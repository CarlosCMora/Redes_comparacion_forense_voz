# ============================================================

# Entrenamiento desde cero 
# ============================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pyAudioAnalysis import audioBasicIO
import Audio_Feature_Extraction as AFE

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Usando GPU:", torch.cuda.get_device_name(0))
else:
    print("Usando CPU")

AUDIO_BASE_PATH = r"\Audios\3 Para evaluar-20251219T020621Z-1-001\3 Para evaluar"  
MODEL_OUT = "speaker_nn.pt"

EPOCHS = 30
LR = 1e-3

random.seed(42)
torch.manual_seed(42)

# ============================================================
# DEFINICIÓN OFICIAL DE PERSONAS (TU TABLA)
# ============================================================

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

# ============================================================
# EXTRACCIÓN MFCC (MISMA QUE app.py)
# ============================================================

def extract_features(audio_path):
    window_size = 0.030
    overlap = 0.015
    VTH_Multiplier = 0.05
    VTH_range = 100

    Fs, x = audioBasicIO.read_audio_file(audio_path)

    if x.ndim > 1:
        x = np.mean(x, axis=1)

    energy = np.square(x)
    voiced_threshold = VTH_Multiplier * np.mean(energy)

    indices = np.arange(0, len(x) - VTH_range, VTH_range)
    sample_means = np.array(
        [np.mean(energy[i:i + VTH_range]) for i in indices]
    )

    valid = indices[sample_means > voiced_threshold]
    if len(valid) == 0:
        return None

    clean = np.concatenate([x[i:i + VTH_range] for i in valid])

    win = int(Fs * window_size)
    step = int(Fs * overlap)

    mfcc = AFE.stFeatureExtraction(clean, Fs, win, step)
    return torch.tensor(mfcc, dtype=torch.float32)

# ============================================================
# DATASET DE PARES
# ============================================================

class PairDataset(Dataset):
    def __init__(self, speaker_to_files, pairs_per_speaker=15):
        self.pairs = []

        speakers = list(speaker_to_files.keys())

        # -------- PARES POSITIVOS --------
        for spk, files in speaker_to_files.items():
            if len(files) < 2:
                continue
            for _ in range(pairs_per_speaker):
                a1, a2 = random.sample(files, 2)
                self.pairs.append((a1, a2, 1))

        # -------- PARES NEGATIVOS --------
        for _ in range(len(self.pairs)):
            spk1, spk2 = random.sample(speakers, 2)
            a1 = random.choice(speaker_to_files[spk1])
            a2 = random.choice(speaker_to_files[spk2])
            self.pairs.append((a1, a2, 0))

        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

# ============================================================
# MODELO NEURONAL
# ============================================================

class FrameEncoder(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        z = self.net(x)
        z = z.mean(dim=0)
        return F.normalize(z, dim=0)


class SimilarityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FrameEncoder()
        self.scorer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        diff = torch.abs(e1 - e2)
        return self.scorer(diff)

# ============================================================
# ENTRENAMIENTO
# ============================================================

def train(model, dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        used = 0

        for a1, a2, label in loader:
            p1 = os.path.join(AUDIO_BASE_PATH, a1[0])
            p2 = os.path.join(AUDIO_BASE_PATH, a2[0])

            f1 = extract_features(p1)
            f2 = extract_features(p2)

            if f1 is None or f2 is None:
                continue

            f1 = f1.to(DEVICE)
            f2 = f2.to(DEVICE)
            y = torch.tensor([label.item()], dtype=torch.float32).to(DEVICE)

            optimizer.zero_grad()
            out = model(f1, f2)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            used += 1

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss / max(1, used):.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nModelo entrenado y guardado como: {MODEL_OUT}")

# ============================================================
# MAIN
# ============================================================


speaker_to_files = {p: [] for p in PERSONAS.keys()}

for fname in os.listdir(AUDIO_BASE_PATH):
    if not fname.lower().endswith(".wav"):
        continue
    for persona in PERSONAS.keys():
        if fname.startswith(persona):
            speaker_to_files[persona].append(fname)

print("Audios detectados por persona:")
for p, files in speaker_to_files.items():
    print(f"{p}: {len(files)} archivos")

dataset = PairDataset(
    speaker_to_files=speaker_to_files,
    pairs_per_speaker=15
)

model = SimilarityModel().to(DEVICE)
train(model, dataset)
