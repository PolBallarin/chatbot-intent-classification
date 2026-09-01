#!/usr/bin/env python3
"""
Compara los resultados de todos los modelos entrenados.
Genera tabla comparativa, gráficos y matrices de confusión.

Ejecutar después de haber entrenado todos los modelos.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import os
import numpy as np

# Intentar importar matplotlib (opcional para gráficos)
try:
    import matplotlib
    matplotlib.use("Agg")  # Backend sin GUI
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False
    print("⚠️  matplotlib/seaborn no instalados. Se generará solo la tabla sin gráficos.")
    print("   Instalar con: pip install matplotlib seaborn\n")

# ══════════════════════════════════════════════════════════════════════════════
# CARGAR RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

RESULT_FILES = {
    "SVM": "results/svm_results.json",
    "Random Forest": "results/random_forest_results.json",
    "Naive Bayes": "results/naive_bayes_results.json",
    "BiLSTM": "results/bilstm_results.json",
    "BERT": "results/bert_results.json",
}

results = {}
for name, filepath in RESULT_FILES.items():
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            results[name] = json.load(f)

if not results:
    print("❌ No se encontraron resultados. Entrena al menos un modelo primero.")
    exit(1)

print(f"📊 Modelos encontrados: {', '.join(results.keys())}\n")

# ══════════════════════════════════════════════════════════════════════════════
# TABLA COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════

print("═" * 90)
print(f"{'MODELO':<20} {'ACCURACY':>10} {'F1 MACRO':>10} {'F1 WEIGHTED':>12} {'LATENCIA':>12} {'ENTRENO':>12}")
print("═" * 90)

for name, r in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
    acc = f"{r['accuracy']*100:.2f}%"
    f1m = f"{r['f1_macro']*100:.2f}%"
    f1w = f"{r['f1_weighted']*100:.2f}%"
    lat = f"{r['avg_latency_ms']:.3f}ms"
    train = f"{r['train_time_seconds']:.1f}s"
    print(f"{name:<20} {acc:>10} {f1m:>10} {f1w:>12} {lat:>12} {train:>12}")

print("═" * 90)

# Mejor modelo
best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
print(f"\n🏆 Mejor modelo: {best_model[0]} (Accuracy: {best_model[1]['accuracy']*100:.2f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# DETALLE POR INTENT
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'─' * 90}")
print("DETALLE F1-SCORE POR INTENT")
print(f"{'─' * 90}")

# Header
header = f"{'INTENT':<30}"
for name in results:
    header += f" {name:>12}"
print(header)
print("─" * 90)

# Obtener todos los intents
all_intents = set()
for r in results.values():
    all_intents.update(r["per_intent"].keys())

for intent in sorted(all_intents):
    row = f"{intent:<30}"
    for name in results:
        f1 = results[name]["per_intent"].get(intent, {}).get("f1", 0)
        row += f" {f1*100:>11.2f}%"
    print(row)

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

if HAS_PLOTS:
    os.makedirs("results/plots", exist_ok=True)

    # 1. Barplot de accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] * 100 for n in names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    bars = ax.bar(names, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Comparación de Accuracy por Modelo")
    ax.set_ylim(0, 105)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.2f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/plots/accuracy_comparison.png", dpi=150)
    plt.close()

    # 2. Barplot de F1 macro
    fig, ax = plt.subplots(figsize=(10, 6))
    f1s = [results[n]["f1_macro"] * 100 for n in names]
    bars = ax.bar(names, f1s, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("F1-Score Macro (%)")
    ax.set_title("Comparación de F1-Score Macro por Modelo")
    ax.set_ylim(0, 105)

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{f1:.2f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/plots/f1_comparison.png", dpi=150)
    plt.close()

    # 3. Barplot de latencia
    fig, ax = plt.subplots(figsize=(10, 6))
    latencies = [results[n]["avg_latency_ms"] for n in names]
    bars = ax.bar(names, latencies, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Latencia (ms)")
    ax.set_title("Comparación de Latencia por Modelo")

    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(latencies)*0.02,
                f"{lat:.3f}ms", ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/plots/latency_comparison.png", dpi=150)
    plt.close()

    # 4. Matriz de confusión de cada modelo
    for name, r in results.items():
        cm = np.array(r["confusion_matrix"])
        labels = r["labels"]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusión - {name}")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        safe_name = name.lower().replace(" ", "_")
        plt.savefig(f"results/plots/confusion_{safe_name}.png", dpi=150)
        plt.close()

    # 5. F1 por intent (heatmap)
    intent_list = sorted(all_intents)
    model_names = list(results.keys())
    f1_matrix = []

    for intent in intent_list:
        row = []
        for name in model_names:
            f1 = results[name]["per_intent"].get(intent, {}).get("f1", 0) * 100
            row.append(f1)
        f1_matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(f1_matrix, annot=True, fmt=".1f", cmap="YlGn",
                xticklabels=model_names, yticklabels=intent_list, ax=ax)
    ax.set_title("F1-Score (%) por Intent y Modelo")
    plt.tight_layout()
    plt.savefig("results/plots/f1_per_intent_heatmap.png", dpi=150)
    plt.close()

    print(f"\n📈 Gráficos guardados en results/plots/")

# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR RESUMEN
# ══════════════════════════════════════════════════════════════════════════════

summary = {
    "best_model": best_model[0],
    "best_accuracy": best_model[1]["accuracy"],
    "models": {
        name: {
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "f1_weighted": r["f1_weighted"],
            "avg_latency_ms": r["avg_latency_ms"],
            "train_time_seconds": r["train_time_seconds"],
        }
        for name, r in results.items()
    },
}

with open("results/comparison_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("✅ Resumen guardado en results/comparison_summary.json")
