#!/usr/bin/env python3
"""
Entrenamiento de BiLSTM para clasificación de intents.
Modelo deep learning recurrente. Usa GPU si está disponible.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 50
MIN_WORD_FREQ = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Dispositivo: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ══════════════════════════════════════════════════════════════════════════════
# CARGAR DATOS
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(filepath):
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
            labels.append(data["intent"])
    return texts, labels

print("\n📂 Cargando dataset...")
train_texts, train_labels = load_jsonl("dataset/train.jsonl")
val_texts, val_labels = load_jsonl("dataset/val.jsonl")
test_texts, test_labels = load_jsonl("dataset/test.jsonl")

print(f"  Train: {len(train_texts):,} ejemplos")
print(f"  Val:   {len(val_texts):,} ejemplos")
print(f"  Test:  {len(test_texts):,} ejemplos")

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

print("\n🔧 Preprocesando...")

# Codificar labels
le = LabelEncoder()
le.fit(train_labels + val_labels + test_labels)
num_classes = len(le.classes_)
print(f"  Clases: {num_classes}")

# Construir vocabulario desde train
def tokenize(text):
    return text.lower().split()

word_counts = Counter()
for text in train_texts:
    word_counts.update(tokenize(text))

# Vocabulario: solo palabras con frecuencia >= MIN_WORD_FREQ
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, count in word_counts.items():
    if count >= MIN_WORD_FREQ:
        vocab[word] = len(vocab)

print(f"  Vocabulario: {len(vocab):,} palabras")

# Convertir texto a secuencia de índices
def text_to_indices(text, vocab, max_len):
    tokens = tokenize(text)[:max_len]
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    # Padding
    indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return indices


# ══════════════════════════════════════════════════════════════════════════════
# DATASET Y DATALOADER
# ══════════════════════════════════════════════════════════════════════════════

class IntentDataset(Dataset):
    def __init__(self, texts, labels, vocab, label_encoder, max_len):
        self.sequences = [text_to_indices(t, vocab, max_len) for t in texts]
        self.labels = label_encoder.transform(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

train_dataset = IntentDataset(train_texts, train_labels, vocab, le, MAX_SEQ_LEN)
val_dataset = IntentDataset(val_texts, val_labels, vocab, le, MAX_SEQ_LEN)
test_dataset = IntentDataset(test_texts, test_labels, vocab, le, MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ══════════════════════════════════════════════════════════════════════════════
# MODELO
# ══════════════════════════════════════════════════════════════════════════════

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 por bidireccional

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Concatenar último hidden de ambas direcciones
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

model = BiLSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=num_classes,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parámetros del modelo: {total_params:,}")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

print(f"\n🚀 Entrenando BiLSTM ({EPOCHS} epochs)...")
start_time = time.time()
best_val_acc = 0
best_epoch = 0

for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    train_acc = correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(batch_y).sum().item()
            val_total += batch_y.size(0)

    val_acc = val_correct / val_total
    scheduler.step(1 - val_acc)

    print(f"  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/len(train_loader):.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # Guardar mejor modelo
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save({
            "model_state": model.state_dict(),
            "vocab": vocab,
            "label_encoder_classes": le.classes_.tolist(),
            "config": {
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_classes": num_classes,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "max_seq_len": MAX_SEQ_LEN,
            },
        }, "models/bilstm_model.pt")

train_time = time.time() - start_time
print(f"\n  Mejor epoch: {best_epoch} (val_acc={best_val_acc:.4f})")
print(f"  Tiempo total: {train_time:.2f}s")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN CON MEJOR MODELO
# ══════════════════════════════════════════════════════════════════════════════

print("\n📊 Evaluando en test set (mejor modelo)...")

# Cargar mejor modelo
checkpoint = torch.load("models/bilstm_model.pt", weights_only=False)
model.load_state_dict(checkpoint["model_state"])
model.eval()

all_preds = []
all_labels = []

start_time = time.time()
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

inference_time = time.time() - start_time
avg_latency = (inference_time / len(test_texts)) * 1000

y_test = np.array(all_labels)
y_pred = np.array(all_preds)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_str = classification_report(y_test, y_pred, target_names=le.classes_)
cm = confusion_matrix(y_test, y_pred)

print(f"\n  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Latencia media: {avg_latency:.3f}ms por ejemplo")
print(f"\n{report_str}")

# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs("results", exist_ok=True)

results = {
    "model": "BiLSTM",
    "accuracy": accuracy,
    "f1_macro": report["macro avg"]["f1-score"],
    "f1_weighted": report["weighted avg"]["f1-score"],
    "train_time_seconds": train_time,
    "avg_latency_ms": avg_latency,
    "best_epoch": best_epoch,
    "total_params": total_params,
    "per_intent": {
        intent: {
            "precision": report[intent]["precision"],
            "recall": report[intent]["recall"],
            "f1": report[intent]["f1-score"],
            "support": report[intent]["support"],
        }
        for intent in le.classes_
    },
    "confusion_matrix": cm.tolist(),
    "labels": le.classes_.tolist(),
}

with open("results/bilstm_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n✅ Modelo guardado en models/bilstm_model.pt")
print("✅ Resultados guardados en results/bilstm_results.json")
