#!/usr/bin/env python3
"""
Entrenamiento de BERT multilingual fine-tuned para clasificación de intents.
Modelo transformer - estado del arte. Usa GPU si está disponible.
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
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "bert-base-multilingual-cased"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 64
WARMUP_RATIO = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Dispositivo: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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

le = LabelEncoder()
le.fit(train_labels + val_labels + test_labels)
num_classes = len(le.classes_)
print(f"  Clases: {num_classes}")

# Cargar tokenizer de BERT
print(f"  Cargando tokenizer: {MODEL_NAME}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_encoder, max_len):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(label_encoder.transform(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }

print("  Tokenizando train...")
train_dataset = IntentDataset(train_texts, train_labels, tokenizer, le, MAX_SEQ_LEN)
print("  Tokenizando val...")
val_dataset = IntentDataset(val_texts, val_labels, tokenizer, le, MAX_SEQ_LEN)
print("  Tokenizando test...")
test_dataset = IntentDataset(test_texts, test_labels, tokenizer, le, MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ══════════════════════════════════════════════════════════════════════════════
# MODELO
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n📦 Cargando modelo: {MODEL_NAME}...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parámetros totales: {total_params:,}")
print(f"  Parámetros entrenables: {trainable_params:,}")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"\n🚀 Entrenando BERT ({EPOCHS} epochs)...")
start_time = time.time()
best_val_acc = 0
best_epoch = 0

for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Progreso cada 100 batches
        if (i + 1) % 100 == 0:
            print(f"    Batch {i+1}/{len(train_loader)} loss={loss.item():.4f}")

    train_acc = correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = outputs.logits.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/len(train_loader):.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # Guardar mejor modelo
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        model.save_pretrained("models/bert_model", safe_serialization=False)
        tokenizer.save_pretrained("models/bert_model")
        # Guardar label encoder
        with open("models/bert_model/label_encoder.json", "w") as f:
            json.dump({"classes": le.classes_.tolist()}, f)

train_time = time.time() - start_time
print(f"\n  Mejor epoch: {best_epoch} (val_acc={best_val_acc:.4f})")
print(f"  Tiempo total: {train_time:.2f}s")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN CON MEJOR MODELO
# ══════════════════════════════════════════════════════════════════════════════

print("\n📊 Evaluando en test set (mejor modelo)...")

# Cargar mejor modelo
model = BertForSequenceClassification.from_pretrained("models/bert_model")
model.to(device)
model.eval()

all_preds = []
all_labels = []

start_time = time.time()
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = outputs.logits.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch["labels"].numpy())

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
    "model": "BERT multilingual",
    "base_model": MODEL_NAME,
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

with open("results/bert_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n✅ Modelo guardado en models/bert_model/")
print("✅ Resultados guardados en results/bert_results.json")
