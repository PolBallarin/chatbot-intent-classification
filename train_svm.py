#!/usr/bin/env python3
"""
Entrenamiento de SVM + TF-IDF para clasificación de intents.
Modelo baseline clásico.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import time
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

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

print("📂 Cargando dataset...")
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
y_train = le.transform(train_labels)
y_val = le.transform(val_labels)
y_test = le.transform(test_labels)

# TF-IDF: convertir texto a vectores numéricos
# Usa unigramas y bigramas para capturar patrones como "lista compra", "qué toma"
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents=None,  # Mantener acentos (importante para español)
)

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

print(f"  Vocabulario: {len(vectorizer.vocabulary_):,} features")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

print("\n🚀 Entrenando SVM...")
start_time = time.time()

model = LinearSVC(
    C=1.0,
    max_iter=10000,
    class_weight="balanced",  # Compensar intents con menos ejemplos
)
model.fit(X_train, y_train)

train_time = time.time() - start_time
print(f"  Tiempo de entrenamiento: {train_time:.2f}s")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

print("\n📊 Evaluando en test set...")

# Medir latencia de inferencia
start_time = time.time()
y_pred = model.predict(X_test)
inference_time = time.time() - start_time
avg_latency = (inference_time / len(test_texts)) * 1000  # ms por ejemplo

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

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Guardar modelo
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump({"model": model, "vectorizer": vectorizer, "label_encoder": le}, f)

# Guardar métricas
results = {
    "model": "SVM + TF-IDF",
    "accuracy": accuracy,
    "f1_macro": report["macro avg"]["f1-score"],
    "f1_weighted": report["weighted avg"]["f1-score"],
    "train_time_seconds": train_time,
    "avg_latency_ms": avg_latency,
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

with open("results/svm_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n✅ Modelo guardado en models/svm_model.pkl")
print("✅ Resultados guardados en results/svm_results.json")
