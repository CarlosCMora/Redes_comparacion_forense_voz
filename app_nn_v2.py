import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier

# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ==========================================================
# CONFIGURACIÓN Y RUTAS 
# ==========================================================
# Detectar GPU Nvidia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# Rutas originales
AUDIO_EVAL_PATH = r"\Audios\3 Para evaluar-20251219T020621Z-1-001\3 Para evaluar"
AUDIO_REF_PATH = r"\Audios\Nuevos audios referencia"
OUTPUT_DIR = "resultados_xvector"

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "resultados_xvector.xlsx")

# Inicializar Excel si no existe
if not os.path.exists(OUTPUT_FILE):
    empty_df = pd.DataFrame()
    empty_df.to_excel(OUTPUT_FILE, index=False)

# ==========================================================
# PERSONAS / SEXO
# ==========================================================
PERSONAS = {
    "Andrea_Itzel": "Mujer",
    "Andrea_Reyes": "Mujer",
    "Dafne": "Mujer",
    "Dalia": "Mujer",
    "Emiliano": "Hombre",
    "Jonathan_Alfaro": "Hombre",
    "Jorge_Ceron": "Hombre",
    "Miguel": "Hombre"
}

# ==========================================================
# MODELO X-VECTOR (Arquitectura basada en el GitHub sugerido)
# ==========================================================
class NeuralSpeakerRecognizer:
    def __init__(self):
        logging.info("Cargando modelo X-Vector (VoxCeleb)...")
        # Usamos SpeechBrain que implementa la arquitectura TDNN (X-Vector)
        
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            run_opts={"device": DEVICE} 
        )
        logging.info("Modelo cargado exitosamente en GPU.")

    def get_embedding(self, audio_path):
        if not os.path.exists(audio_path):
            logging.warning(f"Audio no encontrado: {audio_path}")
            return None
        
        try:
            # Cargar audio y asegurar 16kHz (estándar para X-Vectors)
            signal, fs = torchaudio.load(audio_path)
            
            # Convertir a Mono si es estéreo
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
            
            # Resamplear si no es 16kHz
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000).to(DEVICE)
                signal = signal.to(DEVICE)
                signal = resampler(signal)
            else:
                signal = signal.to(DEVICE)

            # Extraer X-Vector (Embedding)
            with torch.no_grad():
                embeddings = self.classifier.encode_batch(signal)
                # Normalizar vector
                embeddings = F.normalize(embeddings, dim=2)
            
            # Retornar como numpy array plano
            return embeddings.squeeze().cpu().numpy()
            
        except Exception as e:
            logging.error(f"Error procesando {audio_path}: {e}")
            return None

# ==========================================================
# PARSEO DE METADATOS (Copiado de app_nn1.py)
# ==========================================================
def parse_audio_metadata(fname):
    base = os.path.splitext(fname)[0]

    persona_detectada = None
    for persona in PERSONAS.keys():
        if base.startswith(persona):
            persona_detectada = persona
            break

    if persona_detectada is None:
        # Fallback seguro para evitar romper el loop si hay archivos extraños
        return {
            "H": 0, "M": 0, "HESP": 0, "LEC": 0, "T2": 0, "ID": "Desconocido"
        }

    sexo = PERSONAS[persona_detectada]

    H = 1 if sexo == "Hombre" else 0
    M = 1 if sexo == "Mujer" else 0

    HESP = 1 if "HESP" in base else 0
    LEC = 1 if "LEC" in base else 0
    T2 = 1 if "T2" in base else 0

    return {
        "H": H,
        "M": M,
        "HESP": HESP,
        "LEC": LEC,
        "T2": T2,
        "ID": persona_detectada
    }

# ==========================================================
# LÓGICA DE COMPARACIÓN
# ==========================================================
def compute_cosine_similarity(emb1, emb2):
    # Cálculo manual de similitud coseno para arrays numpy 1D
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    return dot_product / (norm1 * norm2)
# ==========================================================
#  FUNCIÓN de  MAPEO
# ==========================================================
def cosine_to_probability(score):
    """
    Convierte la Similitud Coseno (rango -1 a 1) a una pseudo-probabilidad (rango 0 a 1).
    Esto evita LRs negativos y errores en el logaritmo (LLR).
    """
    # Mapeo lineal simple: (-1 se vuelve 0, 1 se vuelve 1)
    prob = (score + 1) / 2
    
    # Aseguramos que nunca sea exactamente 0 para evitar división por cero o log(0)
    # 1e-9 es un número muy pequeño (0.000000001)
    return max(prob, 1e-9)

# ==========================================================
# FUNCIÓN DE COMPARACIÓN 
# ==========================================================

def compara_xvector(audio1_path, audio2_path, recognizer, ref_embeddings_cache, ref_files, dataset_size):
    start = time.time()

    # 1. Obtener embeddings
    emb1 = recognizer.get_embedding(audio1_path)
    emb2 = recognizer.get_embedding(audio2_path)

    if emb1 is None or emb2 is None:
        return None

    # 2. Similitud (Score crudo)
    raw_similitud = float(compute_cosine_similarity(emb1, emb2))
    
    
    # Convertimos el coseno (-1 a 1) en probabilidad (0 a 1)
    Similitud_Prob = cosine_to_probability(raw_similitud)


    # 3. Tipicidad (Promedio contra población de referencia)
    current_refs = ref_files[:dataset_size]
    tip_vals = []

    for rf_name in current_refs:
        rf_path = os.path.join(AUDIO_REF_PATH, rf_name)
        
        if rf_name not in ref_embeddings_cache:
            ref_embeddings_cache[rf_name] = recognizer.get_embedding(rf_path)
        
        emb_r = ref_embeddings_cache[rf_name]
        
        if emb_r is not None:
            # Calculamos el coseno crudo contra la referencia
            raw_ref_score = compute_cosine_similarity(emb1, emb_r)
            
            # Antes de promediar, convertimos a probabilidad
            tip_vals.append(cosine_to_probability(raw_ref_score))

    # Promedio de probabilidades (Tipicidad)
    Tipicidad_Prob = float(np.mean(tip_vals)) if len(tip_vals) > 0 else 1e-9

    # 4. Cálculo de LR y LLR 
    LR = Similitud_Prob / Tipicidad_Prob
    
    # LLR (Log Likelihood Ratio)
    LLR = float(np.log(LR))

    elapsed = time.time() - start
    return raw_similitud, Tipicidad_Prob, LR, LLR, elapsed
# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    
    # 1. Instanciar el modelo X-Vector
    recognizer = NeuralSpeakerRecognizer()

    # 2. Listar archivos
    ref_files = sorted([p.name for p in Path(AUDIO_REF_PATH).glob("*.wav")])
    
    # Cache para embeddings de referencia (optimización crítica)
    ref_embeddings_cache = {}

    # Lista de pares 
    audio_pairs = [
        ("Andrea_Itzel_HESP.wav", "Andrea_Itzel_HESP_T2.wav","R1"),
        ("Andrea_Itzel_HESP.wav", "Andrea_Itzel_LEC.wav","R2"),
        ("Andrea_Itzel_HESP.wav", "Andrea_Itzel_LEC_T2.wav","R3"),
        ("Andrea_Itzel_HESP.wav", "Andrea_Reyes_HESP.wav","R4"),
        ("Andrea_Itzel_HESP.wav", "Andrea_Reyes_HESP_T2.wav","R5"),
        ("Andrea_Itzel_HESP.wav", "Andrea_Reyes_LEC.wav","R6"),
        ("Andrea_Itzel_HESP.wav", "Andrea_Reyes_LEC_T2.wav","R7"),
        ("Andrea_Itzel_HESP.wav", "Dafne_HESP.wav","R8"),
        ("Andrea_Itzel_HESP.wav", "Dafne_HESP_T2.wav","R9"),
        ("Andrea_Itzel_HESP.wav", "Dafne_LEC.wav","R10"),
        ("Andrea_Itzel_HESP.wav", "Dafne_LEC_T2.wav","R11"),
        ("Andrea_Itzel_HESP.wav", "Dalia_HESP.wav","R12"),
        ("Andrea_Itzel_HESP.wav", "Dalia_HESP_T2.wav","R13"),
        ("Andrea_Itzel_HESP.wav", "Dalia_LEC.wav","R14"),
        ("Andrea_Itzel_HESP.wav", "Dalia_LEC_T2.wav","R15"),
        ("Andrea_Itzel_HESP.wav", "Emiliano_HESP.wav","R16"),
        ("Andrea_Itzel_HESP.wav", "Emiliano_HESP_T2.wav","R17"),
        ("Andrea_Itzel_HESP.wav", "Emiliano_LEC.wav","R18"),
        ("Andrea_Itzel_HESP.wav", "Emiliano_LEC_T2.wav","R19"),
        ("Andrea_Itzel_HESP.wav", "Jonathan_Alfaro_HESP.wav","R20"),
        ("Andrea_Itzel_HESP.wav", "Jonathan_Alfaro_HESP_T2.wav","R21"),
        ("Andrea_Itzel_HESP.wav", "Jonathan_Alfaro_LEC.wav","R22"),
        ("Andrea_Itzel_HESP.wav", "Jonathan_Alfaro_LEC_T2.wav","R23"),
        ("Andrea_Itzel_HESP.wav", "Jorge_Ceron_HESP.wav","R24"),
        ("Andrea_Itzel_HESP.wav", "Jorge_Ceron_HESP_T2.wav","R25"),
        ("Andrea_Itzel_HESP.wav", "Jorge_Ceron_LEC.wav","R26"),
        ("Andrea_Itzel_HESP.wav", "Jorge_Ceron_LEC_T2.wav","R27"),
        ("Andrea_Itzel_HESP.wav", "Miguel_HESP.wav","R28"),
        ("Andrea_Itzel_HESP.wav", "Miguel_HESP_T2.wav","R29"),
        ("Andrea_Itzel_HESP.wav", "Miguel_LEC.wav","R30"),
        ("Andrea_Itzel_HESP.wav", "Miguel_LEC_T2.wav","R31"),
        ("Andrea_Itzel_HESP_T2.wav", "Andrea_Itzel_LEC.wav","R32"),
        ("Andrea_Itzel_HESP_T2.wav", "Andrea_Itzel_LEC_T2.wav","R33"),
        ("Andrea_Itzel_HESP_T2.wav", "Andrea_Reyes_HESP.wav","R34"),
        ("Andrea_Itzel_HESP_T2.wav", "Andrea_Reyes_HESP_T2.wav","R35"),
        ("Andrea_Itzel_HESP_T2.wav", "Andrea_Reyes_LEC.wav","R36"),
        ("Andrea_Itzel_HESP_T2.wav", "Andrea_Reyes_LEC_T2.wav","R37"),
        ("Andrea_Itzel_HESP_T2.wav", "Dafne_HESP.wav","R38"),
        ("Andrea_Itzel_HESP_T2.wav", "Dafne_HESP_T2.wav","R39"),
        ("Andrea_Itzel_HESP_T2.wav", "Dafne_LEC.wav","R40"),
        ("Andrea_Itzel_HESP_T2.wav", "Dafne_LEC_T2.wav","R41"),
        ("Andrea_Itzel_HESP_T2.wav", "Dalia_HESP.wav","R42"),
        ("Andrea_Itzel_HESP_T2.wav", "Dalia_HESP_T2.wav","R43"),
        ("Andrea_Itzel_HESP_T2.wav", "Dalia_LEC.wav","R44"),
        ("Andrea_Itzel_HESP_T2.wav", "Dalia_LEC_T2.wav","R45"),
        ("Andrea_Itzel_HESP_T2.wav", "Emiliano_HESP.wav","R46"),
        ("Andrea_Itzel_HESP_T2.wav", "Emiliano_HESP_T2.wav","R47"),
        ("Andrea_Itzel_HESP_T2.wav", "Emiliano_LEC.wav","R48"),
        ("Andrea_Itzel_HESP_T2.wav", "Emiliano_LEC_T2.wav","R49"),
        ("Andrea_Itzel_HESP_T2.wav", "Jonathan_Alfaro_HESP.wav","R50"),
        ("Andrea_Itzel_HESP_T2.wav", "Jonathan_Alfaro_HESP_T2.wav","R51"),
        ("Andrea_Itzel_HESP_T2.wav", "Jonathan_Alfaro_LEC.wav","R52"),
        ("Andrea_Itzel_HESP_T2.wav", "Jonathan_Alfaro_LEC_T2.wav","R53"),
        ("Andrea_Itzel_HESP_T2.wav", "Jorge_Ceron_HESP.wav","R54"),
        ("Andrea_Itzel_HESP_T2.wav", "Jorge_Ceron_HESP_T2.wav","R55"),
        ("Andrea_Itzel_HESP_T2.wav", "Jorge_Ceron_LEC.wav","R56"),
        ("Andrea_Itzel_HESP_T2.wav", "Jorge_Ceron_LEC_T2.wav","R57"),
        ("Andrea_Itzel_HESP_T2.wav", "Miguel_HESP.wav","R58"),
        ("Andrea_Itzel_HESP_T2.wav", "Miguel_HESP_T2.wav","R59"),
        ("Andrea_Itzel_HESP_T2.wav", "Miguel_LEC.wav","R60"),
        ("Andrea_Itzel_HESP_T2.wav", "Miguel_LEC_T2.wav","R61"),
        ("Andrea_Itzel_LEC.wav", "Andrea_Itzel_LEC_T2.wav","R62"),
        ("Andrea_Itzel_LEC.wav", "Andrea_Reyes_HESP.wav","R63"),
        ("Andrea_Itzel_LEC.wav", "Andrea_Reyes_HESP_T2.wav","R64"),
        ("Andrea_Itzel_LEC.wav", "Andrea_Reyes_LEC.wav","R65"),
        ("Andrea_Itzel_LEC.wav", "Andrea_Reyes_LEC_T2.wav","R66"),
        ("Andrea_Itzel_LEC.wav", "Dafne_HESP.wav","R67"),
        ("Andrea_Itzel_LEC.wav", "Dafne_HESP_T2.wav","R68"),
        ("Andrea_Itzel_LEC.wav", "Dafne_LEC.wav","R69"),
        ("Andrea_Itzel_LEC.wav", "Dafne_LEC_T2.wav","R70"),
        ("Andrea_Itzel_LEC.wav", "Dalia_HESP.wav","R71"),
        ("Andrea_Itzel_LEC.wav", "Dalia_HESP_T2.wav","R72"),
        ("Andrea_Itzel_LEC.wav", "Dalia_LEC.wav","R73"),
        ("Andrea_Itzel_LEC.wav", "Dalia_LEC_T2.wav","R74"),
        ("Andrea_Itzel_LEC.wav", "Emiliano_HESP.wav","R75"),
        ("Andrea_Itzel_LEC.wav", "Emiliano_HESP_T2.wav","R76"),
        ("Andrea_Itzel_LEC.wav", "Emiliano_LEC.wav","R77"),
        ("Andrea_Itzel_LEC.wav", "Emiliano_LEC_T2.wav","R78"),
        ("Andrea_Itzel_LEC.wav", "Jonathan_Alfaro_HESP.wav","R79"),
        ("Andrea_Itzel_LEC.wav", "Jonathan_Alfaro_HESP_T2.wav","R80"),
        ("Andrea_Itzel_LEC.wav", "Jonathan_Alfaro_LEC.wav","R81"),
        ("Andrea_Itzel_LEC.wav", "Jonathan_Alfaro_LEC_T2.wav","R82"),
        ("Andrea_Itzel_LEC.wav", "Jorge_Ceron_HESP.wav","R83"),
        ("Andrea_Itzel_LEC.wav", "Jorge_Ceron_HESP_T2.wav","R84"),
        ("Andrea_Itzel_LEC.wav", "Jorge_Ceron_LEC.wav","R85"),
        ("Andrea_Itzel_LEC.wav", "Jorge_Ceron_LEC_T2.wav","R86"),
        ("Andrea_Itzel_LEC.wav", "Miguel_HESP.wav","R87"),
        ("Andrea_Itzel_LEC.wav", "Miguel_HESP_T2.wav","R88"),
        ("Andrea_Itzel_LEC.wav", "Miguel_LEC.wav","R89"),
        ("Andrea_Itzel_LEC.wav", "Miguel_LEC_T2.wav","R90"),
        ("Andrea_Itzel_LEC_T2.wav", "Andrea_Reyes_HESP.wav","R91"),
        ("Andrea_Itzel_LEC_T2.wav", "Andrea_Reyes_HESP_T2.wav","R92"),
        ("Andrea_Itzel_LEC_T2.wav", "Andrea_Reyes_LEC.wav","R93"),
        ("Andrea_Itzel_LEC_T2.wav", "Andrea_Reyes_LEC_T2.wav","R94"),
        ("Andrea_Itzel_LEC_T2.wav", "Dafne_HESP.wav","R95"),
        ("Andrea_Itzel_LEC_T2.wav", "Dafne_HESP_T2.wav","R96"),
        ("Andrea_Itzel_LEC_T2.wav", "Dafne_LEC.wav","R97"),
        ("Andrea_Itzel_LEC_T2.wav", "Dafne_LEC_T2.wav","R98"),
        ("Andrea_Itzel_LEC_T2.wav", "Dalia_HESP.wav","R99"),
        ("Andrea_Itzel_LEC_T2.wav", "Dalia_HESP_T2.wav","R100"),
        ("Andrea_Itzel_LEC_T2.wav", "Dalia_LEC.wav","R101"),
        ("Andrea_Itzel_LEC_T2.wav", "Dalia_LEC_T2.wav","R102"),
        ("Andrea_Itzel_LEC_T2.wav", "Emiliano_HESP.wav","R103"),
        ("Andrea_Itzel_LEC_T2.wav", "Emiliano_HESP_T2.wav","R104"),
        ("Andrea_Itzel_LEC_T2.wav", "Emiliano_LEC.wav","R105"),
        ("Andrea_Itzel_LEC_T2.wav", "Emiliano_LEC_T2.wav","R106"),
        ("Andrea_Itzel_LEC_T2.wav", "Jonathan_Alfaro_HESP.wav","R107"),
        ("Andrea_Itzel_LEC_T2.wav", "Jonathan_Alfaro_HESP_T2.wav","R108"),
        ("Andrea_Itzel_LEC_T2.wav", "Jonathan_Alfaro_LEC.wav","R109"),
        ("Andrea_Itzel_LEC_T2.wav", "Jonathan_Alfaro_LEC_T2.wav","R110"),
        ("Andrea_Itzel_LEC_T2.wav", "Jorge_Ceron_HESP.wav","R111"),
        ("Andrea_Itzel_LEC_T2.wav", "Jorge_Ceron_HESP_T2.wav","R112"),
        ("Andrea_Itzel_LEC_T2.wav", "Jorge_Ceron_LEC.wav","R113"),
        ("Andrea_Itzel_LEC_T2.wav", "Jorge_Ceron_LEC_T2.wav","R114"),
        ("Andrea_Itzel_LEC_T2.wav", "Miguel_HESP.wav","R115"),
        ("Andrea_Itzel_LEC_T2.wav", "Miguel_HESP_T2.wav","R116"),
        ("Andrea_Itzel_LEC_T2.wav", "Miguel_LEC.wav","R117"),
        ("Andrea_Itzel_LEC_T2.wav", "Miguel_LEC_T2.wav","R118"),
        ("Andrea_Reyes_HESP.wav", "Andrea_Reyes_HESP_T2.wav","R119"),
        ("Andrea_Reyes_HESP.wav", "Andrea_Reyes_LEC.wav","R120"),
        ("Andrea_Reyes_HESP.wav", "Andrea_Reyes_LEC_T2.wav","R121"),
        ("Andrea_Reyes_HESP.wav", "Dafne_HESP.wav","R122"),
        ("Andrea_Reyes_HESP.wav", "Dafne_HESP_T2.wav","R123"),
        ("Andrea_Reyes_HESP.wav", "Dafne_LEC.wav","R124"),
        ("Andrea_Reyes_HESP.wav", "Dafne_LEC_T2.wav","R125"),
        ("Andrea_Reyes_HESP.wav", "Dalia_HESP.wav","R126"),
        ("Andrea_Reyes_HESP.wav", "Dalia_HESP_T2.wav","R127"),
        ("Andrea_Reyes_HESP.wav", "Dalia_LEC.wav","R128"),
        ("Andrea_Reyes_HESP.wav", "Dalia_LEC_T2.wav","R129"),
        ("Andrea_Reyes_HESP.wav", "Emiliano_HESP.wav","R130"),
        ("Andrea_Reyes_HESP.wav", "Emiliano_HESP_T2.wav","R131"),
        ("Andrea_Reyes_HESP.wav", "Emiliano_LEC.wav","R132"),
        ("Andrea_Reyes_HESP.wav", "Emiliano_LEC_T2.wav","R133"),
        ("Andrea_Reyes_HESP.wav", "Jonathan_Alfaro_HESP.wav","R134"),
        ("Andrea_Reyes_HESP.wav", "Jonathan_Alfaro_HESP_T2.wav","R135"),
        ("Andrea_Reyes_HESP.wav", "Jonathan_Alfaro_LEC.wav","R136"),
        ("Andrea_Reyes_HESP.wav", "Jonathan_Alfaro_LEC_T2.wav","R137"),
        ("Andrea_Reyes_HESP.wav", "Jorge_Ceron_HESP.wav","R138"),
        ("Andrea_Reyes_HESP.wav", "Jorge_Ceron_HESP_T2.wav","R139"),
        ("Andrea_Reyes_HESP.wav", "Jorge_Ceron_LEC.wav","R140"),
        ("Andrea_Reyes_HESP.wav", "Jorge_Ceron_LEC_T2.wav","R141"),
        ("Andrea_Reyes_HESP.wav", "Miguel_HESP.wav","R142"),
        ("Andrea_Reyes_HESP.wav", "Miguel_HESP_T2.wav","R143"),
        ("Andrea_Reyes_HESP.wav", "Miguel_LEC.wav","R144"),
        ("Andrea_Reyes_HESP.wav", "Miguel_LEC_T2.wav","R145"),
        ("Andrea_Reyes_HESP_T2.wav", "Andrea_Reyes_LEC.wav","R146"),
        ("Andrea_Reyes_HESP_T2.wav", "Andrea_Reyes_LEC_T2.wav","R147"),
        ("Andrea_Reyes_HESP_T2.wav", "Dafne_HESP.wav","R148"),
        ("Andrea_Reyes_HESP_T2.wav", "Dafne_HESP_T2.wav","R149"),
        ("Andrea_Reyes_HESP_T2.wav", "Dafne_LEC.wav","R150"),
        ("Andrea_Reyes_HESP_T2.wav", "Dafne_LEC_T2.wav","R151"),
        ("Andrea_Reyes_HESP_T2.wav", "Dalia_HESP.wav","R152"),
        ("Andrea_Reyes_HESP_T2.wav", "Dalia_HESP_T2.wav","R153"),
        ("Andrea_Reyes_HESP_T2.wav", "Dalia_LEC.wav","R154"),
        ("Andrea_Reyes_HESP_T2.wav", "Dalia_LEC_T2.wav","R155"),
        ("Andrea_Reyes_HESP_T2.wav", "Emiliano_HESP.wav","R156"),
        ("Andrea_Reyes_HESP_T2.wav", "Emiliano_HESP_T2.wav","R157"),
        ("Andrea_Reyes_HESP_T2.wav", "Emiliano_LEC.wav","R158"),
        ("Andrea_Reyes_HESP_T2.wav", "Emiliano_LEC_T2.wav","R159"),
        ("Andrea_Reyes_HESP_T2.wav", "Jonathan_Alfaro_HESP.wav","R160"),
        ("Andrea_Reyes_HESP_T2.wav", "Jonathan_Alfaro_HESP_T2.wav","R161"),
        ("Andrea_Reyes_HESP_T2.wav", "Jonathan_Alfaro_LEC.wav","R162"),
        ("Andrea_Reyes_HESP_T2.wav", "Jonathan_Alfaro_LEC_T2.wav","R163"),
        ("Andrea_Reyes_HESP_T2.wav", "Jorge_Ceron_HESP.wav","R164"),
        ("Andrea_Reyes_HESP_T2.wav", "Jorge_Ceron_HESP_T2.wav","R165"),
        ("Andrea_Reyes_HESP_T2.wav", "Jorge_Ceron_LEC.wav","R166"),
        ("Andrea_Reyes_HESP_T2.wav", "Jorge_Ceron_LEC_T2.wav","R167"),
        ("Andrea_Reyes_HESP_T2.wav", "Miguel_HESP.wav","R168"),
        ("Andrea_Reyes_HESP_T2.wav", "Miguel_HESP_T2.wav","R169"),
        ("Andrea_Reyes_HESP_T2.wav", "Miguel_LEC.wav","R170"),
        ("Andrea_Reyes_HESP_T2.wav", "Miguel_LEC_T2.wav","R171"),
        ("Andrea_Reyes_LEC.wav", "Andrea_Reyes_LEC_T2.wav","R172"),
        ("Andrea_Reyes_LEC.wav", "Dafne_HESP.wav","R173"),
        ("Andrea_Reyes_LEC.wav", "Dafne_HESP_T2.wav","R174"),
        ("Andrea_Reyes_LEC.wav", "Dafne_LEC.wav","R175"),
        ("Andrea_Reyes_LEC.wav", "Dafne_LEC_T2.wav","R176"),
        ("Andrea_Reyes_LEC.wav", "Dalia_HESP.wav","R177"),
        ("Andrea_Reyes_LEC.wav", "Dalia_HESP_T2.wav","R178"),
        ("Andrea_Reyes_LEC.wav", "Dalia_LEC.wav","R179"),
        ("Andrea_Reyes_LEC.wav", "Dalia_LEC_T2.wav","R180"),
        ("Andrea_Reyes_LEC.wav", "Emiliano_HESP.wav","R181"),
        ("Andrea_Reyes_LEC.wav", "Emiliano_HESP_T2.wav","R182"),
        ("Andrea_Reyes_LEC.wav", "Emiliano_LEC.wav","R183"),
        ("Andrea_Reyes_LEC.wav", "Emiliano_LEC_T2.wav","R184"),
        ("Andrea_Reyes_LEC.wav", "Jonathan_Alfaro_HESP.wav","R185"),
        ("Andrea_Reyes_LEC.wav", "Jonathan_Alfaro_HESP_T2.wav","R186"),
        ("Andrea_Reyes_LEC.wav", "Jonathan_Alfaro_LEC.wav","R187"),
        ("Andrea_Reyes_LEC.wav", "Jonathan_Alfaro_LEC_T2.wav","R188"),
        ("Andrea_Reyes_LEC.wav", "Jorge_Ceron_HESP.wav","R189"),
        ("Andrea_Reyes_LEC.wav", "Jorge_Ceron_HESP_T2.wav","R190"),
        ("Andrea_Reyes_LEC.wav", "Jorge_Ceron_LEC.wav","R191"),
        ("Andrea_Reyes_LEC.wav", "Jorge_Ceron_LEC_T2.wav","R192"),
        ("Andrea_Reyes_LEC.wav", "Miguel_HESP.wav","R193"),
        ("Andrea_Reyes_LEC.wav", "Miguel_HESP_T2.wav","R194"),
        ("Andrea_Reyes_LEC.wav", "Miguel_LEC.wav","R195"),
        ("Andrea_Reyes_LEC.wav", "Miguel_LEC_T2.wav","R196"),
        ("Andrea_Reyes_LEC_T2.wav", "Dafne_HESP.wav","R197"),
        ("Andrea_Reyes_LEC_T2.wav", "Dafne_HESP_T2.wav","R198"),
        ("Andrea_Reyes_LEC_T2.wav", "Dafne_LEC.wav","R199"),
        ("Andrea_Reyes_LEC_T2.wav", "Dafne_LEC_T2.wav","R200"),
        ("Andrea_Reyes_LEC_T2.wav", "Dalia_HESP.wav","R201"),
        ("Andrea_Reyes_LEC_T2.wav", "Dalia_HESP_T2.wav","R202"),
        ("Andrea_Reyes_LEC_T2.wav", "Dalia_LEC.wav","R203"),
        ("Andrea_Reyes_LEC_T2.wav", "Dalia_LEC_T2.wav","R204"),
        ("Andrea_Reyes_LEC_T2.wav", "Emiliano_HESP.wav","R205"),
        ("Andrea_Reyes_LEC_T2.wav", "Emiliano_HESP_T2.wav","R206"),
        ("Andrea_Reyes_LEC_T2.wav", "Emiliano_LEC.wav","R207"),
        ("Andrea_Reyes_LEC_T2.wav", "Emiliano_LEC_T2.wav","R208"),
        ("Andrea_Reyes_LEC_T2.wav", "Jonathan_Alfaro_HESP.wav","R209"),
        ("Andrea_Reyes_LEC_T2.wav", "Jonathan_Alfaro_HESP_T2.wav","R210"),
        ("Andrea_Reyes_LEC_T2.wav", "Jonathan_Alfaro_LEC.wav","R211"),
        ("Andrea_Reyes_LEC_T2.wav", "Jonathan_Alfaro_LEC_T2.wav","R212"),
        ("Andrea_Reyes_LEC_T2.wav", "Jorge_Ceron_HESP.wav","R213"),
        ("Andrea_Reyes_LEC_T2.wav", "Jorge_Ceron_HESP_T2.wav","R214"),
        ("Andrea_Reyes_LEC_T2.wav", "Jorge_Ceron_LEC.wav","R215"),
        ("Andrea_Reyes_LEC_T2.wav", "Jorge_Ceron_LEC_T2.wav","R216"),
        ("Andrea_Reyes_LEC_T2.wav", "Miguel_HESP.wav","R217"),
        ("Andrea_Reyes_LEC_T2.wav", "Miguel_HESP_T2.wav","R218"),
        ("Andrea_Reyes_LEC_T2.wav", "Miguel_LEC.wav","R219"),
        ("Andrea_Reyes_LEC_T2.wav", "Miguel_LEC_T2.wav","R220"),
        ("Dafne_HESP.wav", "Dafne_HESP_T2.wav","R221"),
        ("Dafne_HESP.wav", "Dafne_LEC.wav","R222"),
        ("Dafne_HESP.wav", "Dafne_LEC_T2.wav","R223"),
        ("Dafne_HESP.wav", "Dalia_HESP.wav","R224"),
        ("Dafne_HESP.wav", "Dalia_HESP_T2.wav","R225"),
        ("Dafne_HESP.wav", "Dalia_LEC.wav","R226"),
        ("Dafne_HESP.wav", "Dalia_LEC_T2.wav","R227"),
        ("Dafne_HESP.wav", "Emiliano_HESP.wav","R228"),
        ("Dafne_HESP.wav", "Emiliano_HESP_T2.wav","R229"),
        ("Dafne_HESP.wav", "Emiliano_LEC.wav","R230"),
        ("Dafne_HESP.wav", "Emiliano_LEC_T2.wav","R231"),
        ("Dafne_HESP.wav", "Jonathan_Alfaro_HESP.wav","R232"),
        ("Dafne_HESP.wav", "Jonathan_Alfaro_HESP_T2.wav","R233"),
        ("Dafne_HESP.wav", "Jonathan_Alfaro_LEC.wav","R234"),
        ("Dafne_HESP.wav", "Jonathan_Alfaro_LEC_T2.wav","R235"),
        ("Dafne_HESP.wav", "Jorge_Ceron_HESP.wav","R236"),
        ("Dafne_HESP.wav", "Jorge_Ceron_HESP_T2.wav","R237"),
        ("Dafne_HESP.wav", "Jorge_Ceron_LEC.wav","R238"),
        ("Dafne_HESP.wav", "Jorge_Ceron_LEC_T2.wav","R239"),
        ("Dafne_HESP.wav", "Miguel_HESP.wav","R240"),
        ("Dafne_HESP.wav", "Miguel_HESP_T2.wav","R241"),
        ("Dafne_HESP.wav", "Miguel_LEC.wav","R242"),
        ("Dafne_HESP.wav", "Miguel_LEC_T2.wav","R243"),
        ("Dafne_HESP_T2.wav", "Dafne_LEC.wav","R244"),
        ("Dafne_HESP_T2.wav", "Dafne_LEC_T2.wav","R245"),
        ("Dafne_HESP_T2.wav", "Dalia_HESP.wav","R246"),
        ("Dafne_HESP_T2.wav", "Dalia_HESP_T2.wav","R247"),
        ("Dafne_HESP_T2.wav", "Dalia_LEC.wav","R248"),
        ("Dafne_HESP_T2.wav", "Dalia_LEC_T2.wav","R249"),
        ("Dafne_HESP_T2.wav", "Emiliano_HESP.wav","R250"),
        ("Dafne_HESP_T2.wav", "Emiliano_HESP_T2.wav","R251"),
        ("Dafne_HESP_T2.wav", "Emiliano_LEC.wav","R252"),
        ("Dafne_HESP_T2.wav", "Emiliano_LEC_T2.wav","R253"),
        ("Dafne_HESP_T2.wav", "Jonathan_Alfaro_HESP.wav","R254"),
        ("Dafne_HESP_T2.wav", "Jonathan_Alfaro_HESP_T2.wav","R255"),
        ("Dafne_HESP_T2.wav", "Jonathan_Alfaro_LEC.wav","R256"),
        ("Dafne_HESP_T2.wav", "Jonathan_Alfaro_LEC_T2.wav","R257"),
        ("Dafne_HESP_T2.wav", "Jorge_Ceron_HESP.wav","R258"),
        ("Dafne_HESP_T2.wav", "Jorge_Ceron_HESP_T2.wav","R259"),
        ("Dafne_HESP_T2.wav", "Jorge_Ceron_LEC.wav","R260"),
        ("Dafne_HESP_T2.wav", "Jorge_Ceron_LEC_T2.wav","R261"),
        ("Dafne_HESP_T2.wav", "Miguel_HESP.wav","R262"),
        ("Dafne_HESP_T2.wav", "Miguel_HESP_T2.wav","R263"),
        ("Dafne_HESP_T2.wav", "Miguel_LEC.wav","R264"),
        ("Dafne_HESP_T2.wav", "Miguel_LEC_T2.wav","R265"),
        ("Dafne_LEC.wav", "Dafne_LEC_T2.wav","R266"),
        ("Dafne_LEC.wav", "Dalia_HESP.wav","R267"),
        ("Dafne_LEC.wav", "Dalia_HESP_T2.wav","R268"),
        ("Dafne_LEC.wav", "Dalia_LEC.wav","R269"),
        ("Dafne_LEC.wav", "Dalia_LEC_T2.wav","R270"),
        ("Dafne_LEC.wav", "Emiliano_HESP.wav","R271"),
        ("Dafne_LEC.wav", "Emiliano_HESP_T2.wav","R272"),
        ("Dafne_LEC.wav", "Emiliano_LEC.wav","R273"),
        ("Dafne_LEC.wav", "Emiliano_LEC_T2.wav","R274"),
        ("Dafne_LEC.wav", "Jonathan_Alfaro_HESP.wav","R275"),
        ("Dafne_LEC.wav", "Jonathan_Alfaro_HESP_T2.wav","R276"),
        ("Dafne_LEC.wav", "Jonathan_Alfaro_LEC.wav","R277"),
        ("Dafne_LEC.wav", "Jonathan_Alfaro_LEC_T2.wav","R278"),
        ("Dafne_LEC.wav", "Jorge_Ceron_HESP.wav","R279"),
        ("Dafne_LEC.wav", "Jorge_Ceron_HESP_T2.wav","R280"),
        ("Dafne_LEC.wav", "Jorge_Ceron_LEC.wav","R281"),
        ("Dafne_LEC.wav", "Jorge_Ceron_LEC_T2.wav","R282"),
        ("Dafne_LEC.wav", "Miguel_HESP.wav","R283"),
        ("Dafne_LEC.wav", "Miguel_HESP_T2.wav","R284"),
        ("Dafne_LEC.wav", "Miguel_LEC.wav","R285"),
        ("Dafne_LEC.wav", "Miguel_LEC_T2.wav","R286"),
        ("Dafne_LEC_T2.wav", "Dalia_HESP.wav","R287"),
        ("Dafne_LEC_T2.wav", "Dalia_HESP_T2.wav","R288"),
        ("Dafne_LEC_T2.wav", "Dalia_LEC.wav","R289"),
        ("Dafne_LEC_T2.wav", "Dalia_LEC_T2.wav","R290"),
        ("Dafne_LEC_T2.wav", "Emiliano_HESP.wav","R291"),
        ("Dafne_LEC_T2.wav", "Emiliano_HESP_T2.wav","R292"),
        ("Dafne_LEC_T2.wav", "Emiliano_LEC.wav","R293"),
        ("Dafne_LEC_T2.wav", "Emiliano_LEC_T2.wav","R294"),
        ("Dafne_LEC_T2.wav", "Jonathan_Alfaro_HESP.wav","R295"),
        ("Dafne_LEC_T2.wav", "Jonathan_Alfaro_HESP_T2.wav","R296"),
        ("Dafne_LEC_T2.wav", "Jonathan_Alfaro_LEC.wav","R297"),
        ("Dafne_LEC_T2.wav", "Jonathan_Alfaro_LEC_T2.wav","R298"),
        ("Dafne_LEC_T2.wav", "Jorge_Ceron_HESP.wav","R299"),
        ("Dafne_LEC_T2.wav", "Jorge_Ceron_HESP_T2.wav","R300"),
        ("Dafne_LEC_T2.wav", "Jorge_Ceron_LEC.wav","R301"),
        ("Dafne_LEC_T2.wav", "Jorge_Ceron_LEC_T2.wav","R302"),
        ("Dafne_LEC_T2.wav", "Miguel_HESP.wav","R303"),
        ("Dafne_LEC_T2.wav", "Miguel_HESP_T2.wav","R304"),
        ("Dafne_LEC_T2.wav", "Miguel_LEC.wav","R305"),
        ("Dafne_LEC_T2.wav", "Miguel_LEC_T2.wav","R306"),
        ("Dalia_HESP.wav", "Dalia_HESP_T2.wav","R307"),
        ("Dalia_HESP.wav", "Dalia_LEC.wav","R308"),
        ("Dalia_HESP.wav", "Dalia_LEC_T2.wav","R309"),
        ("Dalia_HESP.wav", "Emiliano_HESP.wav","R310"),
        ("Dalia_HESP.wav", "Emiliano_HESP_T2.wav","R311"),
        ("Dalia_HESP.wav", "Emiliano_LEC.wav","R312"),
        ("Dalia_HESP.wav", "Emiliano_LEC_T2.wav","R313"),
        ("Dalia_HESP.wav", "Jonathan_Alfaro_HESP.wav","R314"),
        ("Dalia_HESP.wav", "Jonathan_Alfaro_HESP_T2.wav","R315"),
        ("Dalia_HESP.wav", "Jonathan_Alfaro_LEC.wav","R316"),
        ("Dalia_HESP.wav", "Jonathan_Alfaro_LEC_T2.wav","R317"),
        ("Dalia_HESP.wav", "Jorge_Ceron_HESP.wav","R318"),
        ("Dalia_HESP.wav", "Jorge_Ceron_HESP_T2.wav","R319"),
        ("Dalia_HESP.wav", "Jorge_Ceron_LEC.wav","R320"),
        ("Dalia_HESP.wav", "Jorge_Ceron_LEC_T2.wav","R321"),
        ("Dalia_HESP.wav", "Miguel_HESP.wav","R322"),
        ("Dalia_HESP.wav", "Miguel_HESP_T2.wav","R323"),
        ("Dalia_HESP.wav", "Miguel_LEC.wav","R324"),
        ("Dalia_HESP.wav", "Miguel_LEC_T2.wav","R325"),
        ("Dalia_HESP_T2.wav", "Dalia_LEC.wav","R326"),
        ("Dalia_HESP_T2.wav", "Dalia_LEC_T2.wav","R327"),
        ("Dalia_HESP_T2.wav", "Emiliano_HESP.wav","R328"),
        ("Dalia_HESP_T2.wav", "Emiliano_HESP_T2.wav","R329"),
        ("Dalia_HESP_T2.wav", "Emiliano_LEC.wav","R330"),
        ("Dalia_HESP_T2.wav", "Emiliano_LEC_T2.wav","R331"),
        ("Dalia_HESP_T2.wav", "Jonathan_Alfaro_HESP.wav","R332"),
        ("Dalia_HESP_T2.wav", "Jonathan_Alfaro_HESP_T2.wav","R333"),
        ("Dalia_HESP_T2.wav", "Jonathan_Alfaro_LEC.wav","R334"),
        ("Dalia_HESP_T2.wav", "Jonathan_Alfaro_LEC_T2.wav","R335"),
        ("Dalia_HESP_T2.wav", "Jorge_Ceron_HESP.wav","R336"),
        ("Dalia_HESP_T2.wav", "Jorge_Ceron_HESP_T2.wav","R337"),
        ("Dalia_HESP_T2.wav", "Jorge_Ceron_LEC.wav","R338"),
        ("Dalia_HESP_T2.wav", "Jorge_Ceron_LEC_T2.wav","R339"),
        ("Dalia_HESP_T2.wav", "Miguel_HESP.wav","R340"),
        ("Dalia_HESP_T2.wav", "Miguel_HESP_T2.wav","R341"),
        ("Dalia_HESP_T2.wav", "Miguel_LEC.wav","R342"),
        ("Dalia_HESP_T2.wav", "Miguel_LEC_T2.wav","R343"),
        ("Dalia_LEC.wav", "Dalia_LEC_T2.wav","R344"),
        ("Dalia_LEC.wav", "Emiliano_HESP.wav","R345"),
        ("Dalia_LEC.wav", "Emiliano_HESP_T2.wav","R346"),
        ("Dalia_LEC.wav", "Emiliano_LEC.wav","R347"),
        ("Dalia_LEC.wav", "Emiliano_LEC_T2.wav","R348"),
        ("Dalia_LEC.wav", "Jonathan_Alfaro_HESP.wav","R349"),
        ("Dalia_LEC.wav", "Jonathan_Alfaro_HESP_T2.wav","R350"),
        ("Dalia_LEC.wav", "Jonathan_Alfaro_LEC.wav","R351"),
        ("Dalia_LEC.wav", "Jonathan_Alfaro_LEC_T2.wav","R352"),
        ("Dalia_LEC.wav", "Jorge_Ceron_HESP.wav","R353"),
        ("Dalia_LEC.wav", "Jorge_Ceron_HESP_T2.wav","R354"),
        ("Dalia_LEC.wav", "Jorge_Ceron_LEC.wav","R355"),
        ("Dalia_LEC.wav", "Jorge_Ceron_LEC_T2.wav","R356"),
        ("Dalia_LEC.wav", "Miguel_HESP.wav","R357"),
        ("Dalia_LEC.wav", "Miguel_HESP_T2.wav","R358"),
        ("Dalia_LEC.wav", "Miguel_LEC.wav","R359"),
        ("Dalia_LEC.wav", "Miguel_LEC_T2.wav","R360"),
        ("Dalia_LEC_T2.wav", "Emiliano_HESP.wav","R361"),
        ("Dalia_LEC_T2.wav", "Emiliano_HESP_T2.wav","R362"),
        ("Dalia_LEC_T2.wav", "Emiliano_LEC.wav","R363"),
        ("Dalia_LEC_T2.wav", "Emiliano_LEC_T2.wav","R364"),
        ("Dalia_LEC_T2.wav", "Jonathan_Alfaro_HESP.wav","R365"),
        ("Dalia_LEC_T2.wav", "Jonathan_Alfaro_HESP_T2.wav","R366"),
        ("Dalia_LEC_T2.wav", "Jonathan_Alfaro_LEC.wav","R367"),
        ("Dalia_LEC_T2.wav", "Jonathan_Alfaro_LEC_T2.wav","R368"),
        ("Dalia_LEC_T2.wav", "Jorge_Ceron_HESP.wav","R369"),
        ("Dalia_LEC_T2.wav", "Jorge_Ceron_HESP_T2.wav","R370"),
        ("Dalia_LEC_T2.wav", "Jorge_Ceron_LEC.wav","R371"),
        ("Dalia_LEC_T2.wav", "Jorge_Ceron_LEC_T2.wav","R372"),
        ("Dalia_LEC_T2.wav", "Miguel_HESP.wav","R373"),
        ("Dalia_LEC_T2.wav", "Miguel_HESP_T2.wav","R374"),
        ("Dalia_LEC_T2.wav", "Miguel_LEC.wav","R375"),
        ("Dalia_LEC_T2.wav", "Miguel_LEC_T2.wav","R376"),
        ("Emiliano_HESP.wav", "Emiliano_HESP_T2.wav","R377"),
        ("Emiliano_HESP.wav", "Emiliano_LEC.wav","R378"),
        ("Emiliano_HESP.wav", "Emiliano_LEC_T2.wav","R379"),
        ("Emiliano_HESP.wav", "Jonathan_Alfaro_HESP.wav","R380"),
        ("Emiliano_HESP.wav", "Jonathan_Alfaro_HESP_T2.wav","R381"),
        ("Emiliano_HESP.wav", "Jonathan_Alfaro_LEC.wav","R382"),
        ("Emiliano_HESP.wav", "Jonathan_Alfaro_LEC_T2.wav","R383"),
        ("Emiliano_HESP.wav", "Jorge_Ceron_HESP.wav","R384"),
        ("Emiliano_HESP.wav", "Jorge_Ceron_HESP_T2.wav","R385"),
        ("Emiliano_HESP.wav", "Jorge_Ceron_LEC.wav","R386"),
        ("Emiliano_HESP.wav", "Jorge_Ceron_LEC_T2.wav","R387"),
        ("Emiliano_HESP.wav", "Miguel_HESP.wav","R388"),
        ("Emiliano_HESP.wav", "Miguel_HESP_T2.wav","R389"),
        ("Emiliano_HESP.wav", "Miguel_LEC.wav","R390"),
        ("Emiliano_HESP.wav", "Miguel_LEC_T2.wav","R391"),
        ("Emiliano_HESP_T2.wav", "Emiliano_LEC.wav","R392"),
        ("Emiliano_HESP_T2.wav", "Emiliano_LEC_T2.wav","R393"),
        ("Emiliano_HESP_T2.wav", "Jonathan_Alfaro_HESP.wav","R394"),
        ("Emiliano_HESP_T2.wav", "Jonathan_Alfaro_HESP_T2.wav","R395"),
        ("Emiliano_HESP_T2.wav", "Jonathan_Alfaro_LEC.wav","R396"),
        ("Emiliano_HESP_T2.wav", "Jonathan_Alfaro_LEC_T2.wav","R397"),
        ("Emiliano_HESP_T2.wav", "Jorge_Ceron_HESP.wav","R398"),
        ("Emiliano_HESP_T2.wav", "Jorge_Ceron_HESP_T2.wav","R399"),
        ("Emiliano_HESP_T2.wav", "Jorge_Ceron_LEC.wav","R400"),
        ("Emiliano_HESP_T2.wav", "Jorge_Ceron_LEC_T2.wav","R401"),
        ("Emiliano_HESP_T2.wav", "Miguel_HESP.wav","R402"),
        ("Emiliano_HESP_T2.wav", "Miguel_HESP_T2.wav","R403"),
        ("Emiliano_HESP_T2.wav", "Miguel_LEC.wav","R404"),
        ("Emiliano_HESP_T2.wav", "Miguel_LEC_T2.wav","R405"),
        ("Emiliano_LEC.wav", "Emiliano_LEC_T2.wav","R406"),
        ("Emiliano_LEC.wav", "Jonathan_Alfaro_HESP.wav","R407"),
        ("Emiliano_LEC.wav", "Jonathan_Alfaro_HESP_T2.wav","R408"),
        ("Emiliano_LEC.wav", "Jonathan_Alfaro_LEC.wav","R409"),
        ("Emiliano_LEC.wav", "Jonathan_Alfaro_LEC_T2.wav","R410"),
        ("Emiliano_LEC.wav", "Jorge_Ceron_HESP.wav","R411"),
        ("Emiliano_LEC.wav", "Jorge_Ceron_HESP_T2.wav","R412"),
        ("Emiliano_LEC.wav", "Jorge_Ceron_LEC.wav","R413"),
        ("Emiliano_LEC.wav", "Jorge_Ceron_LEC_T2.wav","R414"),
        ("Emiliano_LEC.wav", "Miguel_HESP.wav","R415"),
        ("Emiliano_LEC.wav", "Miguel_HESP_T2.wav","R416"),
        ("Emiliano_LEC.wav", "Miguel_LEC.wav","R417"),
        ("Emiliano_LEC.wav", "Miguel_LEC_T2.wav","R418"),
        ("Emiliano_LEC_T2.wav", "Jonathan_Alfaro_HESP.wav","R419"),
        ("Emiliano_LEC_T2.wav", "Jonathan_Alfaro_HESP_T2.wav","R420"),
        ("Emiliano_LEC_T2.wav", "Jonathan_Alfaro_LEC.wav","R421"),
        ("Emiliano_LEC_T2.wav", "Jonathan_Alfaro_LEC_T2.wav","R422"),
        ("Emiliano_LEC_T2.wav", "Jorge_Ceron_HESP.wav","R423"),
        ("Emiliano_LEC_T2.wav", "Jorge_Ceron_HESP_T2.wav","R424"),
        ("Emiliano_LEC_T2.wav", "Jorge_Ceron_LEC.wav","R425"),
        ("Emiliano_LEC_T2.wav", "Jorge_Ceron_LEC_T2.wav","R426"),
        ("Emiliano_LEC_T2.wav", "Miguel_HESP.wav","R427"),
        ("Emiliano_LEC_T2.wav", "Miguel_HESP_T2.wav","R428"),
        ("Emiliano_LEC_T2.wav", "Miguel_LEC.wav","R429"),
        ("Emiliano_LEC_T2.wav", "Miguel_LEC_T2.wav","R430"),
        ("Jonathan_Alfaro_HESP.wav", "Jonathan_Alfaro_HESP_T2.wav","R431"),
        ("Jonathan_Alfaro_HESP.wav", "Jonathan_Alfaro_LEC.wav","R432"),
        ("Jonathan_Alfaro_HESP.wav", "Jonathan_Alfaro_LEC_T2.wav","R433"),
        ("Jonathan_Alfaro_HESP.wav", "Jorge_Ceron_HESP.wav","R434"),
        ("Jonathan_Alfaro_HESP.wav", "Jorge_Ceron_HESP_T2.wav","R435"),
        ("Jonathan_Alfaro_HESP.wav", "Jorge_Ceron_LEC.wav","R436"),
        ("Jonathan_Alfaro_HESP.wav", "Jorge_Ceron_LEC_T2.wav","R437"),
        ("Jonathan_Alfaro_HESP.wav", "Miguel_HESP.wav","R438"),
        ("Jonathan_Alfaro_HESP.wav", "Miguel_HESP_T2.wav","R439"),
        ("Jonathan_Alfaro_HESP.wav", "Miguel_LEC.wav","R440"),
        ("Jonathan_Alfaro_HESP.wav", "Miguel_LEC_T2.wav","R441"),
        ("Jonathan_Alfaro_HESP_T2.wav", "Jonathan_Alfaro_LEC.wav","R442"),
        ("Jonathan_Alfaro_HESP_T2.wav", "Jonathan_Alfaro_LEC_T2.wav","R443"),
        ("Jonathan_Alfaro_HESP_T2.wav", "Jorge_Ceron_HESP.wav","R444"),
        ("Jonathan_Alfaro_HESP_T2.wav", "Jorge_Ceron_HESP_T2.wav","R445"),
        ("Jonathan_Alfaro_HESP_T2.wav", "Jorge_Ceron_LEC.wav","R446"),
        ("Jonathan_Alfaro_HESP_T2.wav", "Jorge_Ceron_LEC_T2.wav","R447"),
        ("Jonathan_Alfaro_HESP_T2.wav", "Miguel_HESP.wav","R448"),
        ("Jonathan_Alfaro_HESP_T2.wav", "Miguel_HESP_T2.wav","R449"),
        ("Jonathan_Alfaro_HESP_T2.wav", "Miguel_LEC.wav","R450"),
        ("Jonathan_Alfaro_HESP_T2.wav", "Miguel_LEC_T2.wav","R451"),
        ("Jonathan_Alfaro_LEC.wav", "Jonathan_Alfaro_LEC_T2.wav","R452"),
        ("Jonathan_Alfaro_LEC.wav", "Jorge_Ceron_HESP.wav","R453"),
        ("Jonathan_Alfaro_LEC.wav", "Jorge_Ceron_HESP_T2.wav","R454"),
        ("Jonathan_Alfaro_LEC.wav", "Jorge_Ceron_LEC.wav","R455"),
        ("Jonathan_Alfaro_LEC.wav", "Jorge_Ceron_LEC_T2.wav","R456"),
        ("Jonathan_Alfaro_LEC.wav", "Miguel_HESP.wav","R457"),
        ("Jonathan_Alfaro_LEC.wav", "Miguel_HESP_T2.wav","R458"),
        ("Jonathan_Alfaro_LEC.wav", "Miguel_LEC.wav","R459"),
        ("Jonathan_Alfaro_LEC.wav", "Miguel_LEC_T2.wav","R460"),
        ("Jonathan_Alfaro_LEC_T2.wav", "Jorge_Ceron_HESP.wav","R461"),
        ("Jonathan_Alfaro_LEC_T2.wav", "Jorge_Ceron_HESP_T2.wav","R462"),
        ("Jonathan_Alfaro_LEC_T2.wav", "Jorge_Ceron_LEC.wav","R463"),
        ("Jonathan_Alfaro_LEC_T2.wav", "Jorge_Ceron_LEC_T2.wav","R464"),
        ("Jonathan_Alfaro_LEC_T2.wav", "Miguel_HESP.wav","R465"),
        ("Jonathan_Alfaro_LEC_T2.wav", "Miguel_HESP_T2.wav","R466"),
        ("Jonathan_Alfaro_LEC_T2.wav", "Miguel_LEC.wav","R467"),
        ("Jonathan_Alfaro_LEC_T2.wav", "Miguel_LEC_T2.wav","R468"),
        ("Jorge_Ceron_HESP.wav", "Jorge_Ceron_HESP_T2.wav","R469"),
        ("Jorge_Ceron_HESP.wav", "Jorge_Ceron_LEC.wav","R470"),
        ("Jorge_Ceron_HESP.wav", "Jorge_Ceron_LEC_T2.wav","R471"),
        ("Jorge_Ceron_HESP.wav", "Miguel_HESP.wav","R472"),
        ("Jorge_Ceron_HESP.wav", "Miguel_HESP_T2.wav","R473"),
        ("Jorge_Ceron_HESP.wav", "Miguel_LEC.wav","R474"),
        ("Jorge_Ceron_HESP.wav", "Miguel_LEC_T2.wav","R475"),
        ("Jorge_Ceron_HESP_T2.wav", "Jorge_Ceron_LEC.wav","R476"),
        ("Jorge_Ceron_HESP_T2.wav", "Jorge_Ceron_LEC_T2.wav","R477"),
        ("Jorge_Ceron_HESP_T2.wav", "Miguel_HESP.wav","R478"),
        ("Jorge_Ceron_HESP_T2.wav", "Miguel_HESP_T2.wav","R479"),
        ("Jorge_Ceron_HESP_T2.wav", "Miguel_LEC.wav","R480"),
        ("Jorge_Ceron_HESP_T2.wav", "Miguel_LEC_T2.wav","R481"),
        ("Jorge_Ceron_LEC.wav", "Jorge_Ceron_LEC_T2.wav","R482"),
        ("Jorge_Ceron_LEC.wav", "Miguel_HESP.wav","R483"),
        ("Jorge_Ceron_LEC.wav", "Miguel_HESP_T2.wav","R484"),
        ("Jorge_Ceron_LEC.wav", "Miguel_LEC.wav","R485"),
        ("Jorge_Ceron_LEC.wav", "Miguel_LEC_T2.wav","R486"),
        ("Jorge_Ceron_LEC_T2.wav", "Miguel_HESP.wav","R487"),
        ("Jorge_Ceron_LEC_T2.wav", "Miguel_HESP_T2.wav","R488"),
        ("Jorge_Ceron_LEC_T2.wav", "Miguel_LEC.wav","R489"),
        ("Jorge_Ceron_LEC_T2.wav", "Miguel_LEC_T2.wav","R490"),
        ("Miguel_HESP.wav", "Miguel_HESP_T2.wav","R491"),
        ("Miguel_HESP.wav", "Miguel_LEC.wav","R492"),
        ("Miguel_HESP.wav", "Miguel_LEC_T2.wav","R493"),
        ("Miguel_HESP_T2.wav", "Miguel_LEC.wav","R494"),
        ("Miguel_HESP_T2.wav", "Miguel_LEC_T2.wav","R495"),
        ("Miguel_LEC.wav", "Miguel_LEC_T2.wav","R496")  
    ]

    dataset_sizes = [60]

    write_header = True
    completed = 0
    total_tasks = len(audio_pairs) * len(dataset_sizes)
    t0 = time.time()

    logging.info(f"Iniciando evaluación con {total_tasks} comparaciones...")

    for dataset_size in dataset_sizes:
        for a1, a2, nombre in audio_pairs:
            
            # 1. Obtener metadatos desde nombre de archivo
            meta1 = parse_audio_metadata(a1)
            meta2 = parse_audio_metadata(a2)

            same_person = "Misma persona" if meta1["ID"] == meta2["ID"] else "Distinta persona"
            sexo_rel = f"{PERSONAS.get(meta1['ID'], 'UNK')}-{PERSONAS.get(meta2['ID'], 'UNK')}"

            # 2. Ejecutar comparación X-Vector
            result = compara_xvector(
                os.path.join(AUDIO_EVAL_PATH, a1),
                os.path.join(AUDIO_EVAL_PATH, a2),
                recognizer,
                ref_embeddings_cache,
                ref_files,
                dataset_size
            )

            if result is None:
                continue

            similitud, tipicidad, LR, LLR, elapsed = result

            # 3. Construir Fila Excel
            row = {
                "nombre": nombre,
                "dataset_size": dataset_size,
                "Audio 1": a1,
                "Audio 2": a2,

                "H_1": meta1["H"], "M_1": meta1["M"],
                "HESP_1": meta1["HESP"], "LEC_1": meta1["LEC"],
                "T2_1": meta1["T2"], "ID_1": meta1["ID"],

                "H_2": meta2["H"], "M_2": meta2["M"],
                "HESP_2": meta2["HESP"], "LEC_2": meta2["LEC"],
                "T2_2": meta2["T2"], "ID_2": meta2["ID"],

                "d1": similitud, "d2": None, "d3": None,
                "d4": None, "d5": None, "d6": None,

                "diferencia": similitud,
                "Similitud": similitud,
                "Tipicidad": tipicidad,
                "LR": LR,
                "LLR": LLR,

                "misma_persona": same_person,
                "sexo": sexo_rel
            }

            df_row = pd.DataFrame([row])

            # 4. Escribir a Excel en tiempo real
            with pd.ExcelWriter(
                OUTPUT_FILE,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="overlay"
            ) as writer:
                # Si es la primera vez, escribimos en la fila 0, sino al final
                startrow = writer.book.active.max_row if not write_header else 0
                df_row.to_excel(writer, index=False, header=write_header, startrow=startrow)

            write_header = False
            completed += 1

            elapsed_total = time.time() - t0
            eta = (elapsed_total / completed) * (total_tasks - completed)

            if completed % 10 == 0:
                logging.info(
                    f"Tarea {completed}/{total_tasks} | "
                    f"Transcurrido: {elapsed_total:.1f}s | "
                    f"ETA: {eta:.1f}s"
                )

    logging.info("Proceso finalizado correctamente")
    logging.info(f"Resultados guardados en: {OUTPUT_FILE}")