#!/usr/bin/env python3
"""
Convierte el modelo BiLSTM de PyTorch a formato ONNX.
Ejecutar en el PC Windows donde está PyTorch instalado.

Uso: python convert_to_onnx.py

Genera:
  - models/bilstm_model.onnx  (modelo para producción)
  - models/bilstm_vocab.json   (vocabulario + config)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════════════════════
# DEFINICIÓN DEL MODELO (misma que en train_bilstm.py)
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
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# ══════════════════════════════════════════════════════════════════════════════
# CARGAR MODELO
# ══════════════════════════════════════════════════════════════════════════════

print("📂 Cargando modelo BiLSTM...")
checkpoint = torch.load("models/bilstm_model.pt", map_location="cpu", weights_only=False)

config = checkpoint["config"]
vocab = checkpoint["vocab"]
labels = checkpoint["label_encoder_classes"]

model = BiLSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=config["embedding_dim"],
    hidden_dim=config["hidden_dim"],
    num_classes=config["num_classes"],
    num_layers=config["num_layers"],
    dropout=0,  # Sin dropout en inferencia
)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"  Vocabulario: {len(vocab):,} palabras")
print(f"  Clases: {len(labels)}")
print(f"  Config: {config}")

# ══════════════════════════════════════════════════════════════════════════════
# EXPORTAR A ONNX
# ══════════════════════════════════════════════════════════════════════════════

print("\n🔄 Exportando a ONNX...")

# Input de ejemplo (batch de 1, secuencia de max_seq_len)
dummy_input = torch.zeros(1, config["max_seq_len"], dtype=torch.long)

torch.onnx.export(
    model,
    dummy_input,
    "models/bilstm_model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=14,
)

print("  ✅ Guardado en models/bilstm_model.onnx")

# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR VOCABULARIO Y CONFIG
# ══════════════════════════════════════════════════════════════════════════════

vocab_config = {
    "vocab": vocab,
    "labels": labels,
    "max_seq_len": config["max_seq_len"],
}

with open("models/bilstm_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_config, f, ensure_ascii=False)

print("  ✅ Guardado en models/bilstm_vocab.json")

# ══════════════════════════════════════════════════════════════════════════════
# VERIFICAR
# ══════════════════════════════════════════════════════════════════════════════

print("\n🔍 Verificando modelo ONNX...")

try:
    import onnxruntime as ort

    session = ort.InferenceSession("models/bilstm_model.onnx")

    # Test con una frase
    test_text = "ponle ibuprofeno a mi abuela"
    tokens = test_text.lower().split()
    indices = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens]
    indices += [0] * (config["max_seq_len"] - len(indices))
    input_array = [indices]

    import numpy as np
    result = session.run(None, {"input_ids": np.array(input_array, dtype=np.int64)})
    predicted = np.argmax(result[0], axis=1)[0]

    print(f"  Test: \"{test_text}\"")
    print(f"  Predicción: {labels[predicted]}")
    print(f"  ✅ Modelo ONNX funciona correctamente")

    # Tamaño del archivo
    import os
    onnx_size = os.path.getsize("models/bilstm_model.onnx") / (1024 * 1024)
    vocab_size = os.path.getsize("models/bilstm_vocab.json") / (1024 * 1024)
    print(f"\n📦 Tamaños:")
    print(f"  bilstm_model.onnx: {onnx_size:.2f} MB")
    print(f"  bilstm_vocab.json: {vocab_size:.2f} MB")
    print(f"  Total: {onnx_size + vocab_size:.2f} MB")

except ImportError:
    print("  ⚠️  onnxruntime no instalado, no se puede verificar")
    print("  Instalar con: pip install onnxruntime")

print("\n✅ Conversión completada. Copia estos archivos al Mac:")
print("  - models/bilstm_model.onnx")
print("  - models/bilstm_vocab.json")
