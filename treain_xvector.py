import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo de entrenamiento: {DEVICE}")

# Rutas de datos (Ajustadas a su entorno)
AUDIO_BASE_PATH = r"\Audios\3 Para evaluar-20251219T020621Z-1-001\3 Para evaluar"
AUDIO_REF_PATH = r"\Audios\Nuevos audios referencia"
MODEL_OUT = "proyector_siames.pt"

PERSONAS = {
    "Andrea_Itzel": "Mujer", "Andrea_Reyes": "Mujer", "Dafne": "Mujer",
    "Dalia": "Mujer", "Emiliano": "Hombre", "Jonathan_Alfaro": "Hombre",
    "Jorge_Ceron": "Hombre", "Miguel": "Hombre"
}

# ==========================================================
# DEFINICIÓN DE LA ARQUITECTURA
# ==========================================================
class ProyectorSiames(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        # Capa estrictamente lineal para permitir coordenadas negativas
        self.proyeccion = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        z = self.proyeccion(x)
        # Normalización para cálculo directo de coseno
        return F.normalize(z, p=2, dim=-1)

# ==========================================================
# PRE-EXTRACCIÓN DE VECTORES BASE (SPEECHBRAIN)
# ==========================================================
def cargar_vectores_base():
    print("Cargando modelo base SpeechBrain...")
    clasificador = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        run_opts={"device": DEVICE}
    )
    
    vectores_cache = {}
    archivos_por_persona = {p: [] for p in PERSONAS.keys()}
    archivos_ref = []

    # Registrar archivos de evaluación
    for fname in os.listdir(AUDIO_BASE_PATH):
        if not fname.endswith(".wav"): continue
        ruta = os.path.join(AUDIO_BASE_PATH, fname)
        for persona in PERSONAS.keys():
            if fname.startswith(persona):
                archivos_por_persona[persona].append(fname)
                vectores_cache[fname] = procesar_audio(ruta, clasificador)
                break
                
    # Registrar archivos de referencia
    for fname in os.listdir(AUDIO_REF_PATH):
        if not fname.endswith(".wav"): continue
        ruta = os.path.join(AUDIO_REF_PATH, fname)
        archivos_ref.append(fname)
        vectores_cache[fname] = procesar_audio(ruta, clasificador)

    return vectores_cache, archivos_por_persona, archivos_ref

def procesar_audio(ruta, modelo):
    signal, fs = torchaudio.load(ruta)
    if signal.shape[0] > 1: signal = torch.mean(signal, dim=0, keepdim=True)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(fs, 16000)
        signal = resampler(signal)
    with torch.no_grad():
        emb = modelo.encode_batch(signal.to(DEVICE))
        emb = F.normalize(emb, dim=2)
    return emb.squeeze()

# ==========================================================
# GENERACIÓN DE PARES Y ENTRENAMIENTO
# ==========================================================
def entrenar():
    vectores, archivos_por_persona, archivos_ref = cargar_vectores_base()
    
    # Crear pares de entrenamiento
    pares = []
    personas_lista = list(PERSONAS.keys())
    
    # Pares Positivos (Etiqueta 1)
    for p, audios in archivos_por_persona.items():
        for i in range(len(audios)):
            for j in range(i+1, len(audios)):
                pares.append((audios[i], audios[j], 1.0))
                
    # Pares Negativos (Etiqueta -1 para CosineEmbeddingLoss)
    num_positivos = len(pares)
    for _ in range(num_positivos):
        p1, p2 = random.sample(personas_lista, 2)
        a1 = random.choice(archivos_por_persona[p1])
        a2 = random.choice(archivos_por_persona[p2])
        pares.append((a1, a2, -1.0))

    random.shuffle(pares)
    print(f"Total de pares generados para entrenamiento: {len(pares)}")

    # Inicializar Proyector y Optimizador
    modelo = ProyectorSiames().to(DEVICE)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    # Loss específica para forzar el ángulo a 180 grados en pares negativos
    criterio = nn.CosineEmbeddingLoss(margin=0.0)
    
    modelo.train()
    epocas = 50
    batch_size = 64

    for epoca in range(epocas):
        perdida_total = 0
        random.shuffle(pares)
        
        for i in range(0, len(pares), batch_size):
            lote = pares[i:i+batch_size]
            
            vec1 = torch.stack([vectores[p[0]] for p in lote])
            vec2 = torch.stack([vectores[p[1]] for p in lote])
            etiquetas = torch.tensor([p[2] for p in lote], dtype=torch.float32).to(DEVICE)
            
            optimizador.zero_grad()
            
            salida1 = modelo(vec1)
            salida2 = modelo(vec2)
            
            loss = criterio(salida1, salida2, etiquetas)
            loss.backward()
            optimizador.step()
            
            perdida_total += loss.item()
            
        print(f"Época {epoca+1}/{epocas} | Pérdida: {perdida_total/(len(pares)/batch_size):.4f}")

    torch.save(modelo.state_dict(), MODEL_OUT)
    print(f"Proyector entrenado y guardado en: {MODEL_OUT}")

if __name__ == "__main__":
    entrenar()