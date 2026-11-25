"""
train_and_export_guardrail.py
Pipeline completo de entrenamiento, evaluación y exportación ONNX para el Clasificador
Guardrail de Dominio Térmico (`DomainGuardrailNet`) del proyecto Nano-AI.

Flujo de ejecución:
1. Carga el dataset balanceado de 120 frases en inglés (split estratificado 80% Train / 20% Val).
2. Ajusta el vectorizador híbrido TF-IDF (Word + Char N-Grams) únicamente sobre Train (cero data leakage)
   y exporta el vocabulario/IDF + umbral recomendado a `artifacts/guardrail_vocab.json`.
3. Entrena `DomainGuardrailNet` desde cero (pesos aleatorios Kaiming/Xavier) durante 60 épocas
   con `BCELoss` y optimizador `Adam`.
4. Evalúa en el conjunto de validación calculando Accuracy, Precision, Recall, F1-Score y
   Out-of-Domain Rejection Rate (tasa de rechazo de Clase 0) con `threshold = 0.60`.
5. Genera y guarda la gráfica de convergencia y distribución de puntajes en
   `artifacts/guardrail_loss_and_scores.png`.
6. Exporta a `artifacts/guardrail_skeleton.onnx` (`opset_version=17`, `dynamic_axes` en `batch_size`),
   valida con `onnx.checker.check_model` y ejecuta la prueba cruzada en `onnxruntime.InferenceSession`
   sobre 10 frases nuevas en inglés midiendo paridad numérica (< 1e-6) y latencia en milisegundos.

Autor: David Isaí Portillo López (R2 - Model Lead)
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_check import print_diagnostic_report
from src.guardrail.dataset_phrases import (
    UNSEEN_TEST_PHRASES_EN,
    load_or_generate_guardrail_dataset,
)
from src.guardrail.guardrail_model import (
    DomainGuardrailNet,
    fit_tfidf_vocab,
    save_tfidf_vocab,
    summarize_guardrail_footprint,
    vectorize_text_numpy,
)


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
GUARDRAIL_VOCAB_PATH = ARTIFACTS_DIR / "guardrail_vocab.json"
GUARDRAIL_PLOT_PATH = ARTIFACTS_DIR / "guardrail_loss_and_scores.png"
GUARDRAIL_ONNX_PATH = ARTIFACTS_DIR / "guardrail_skeleton.onnx"

RECOMMENDED_THRESHOLD: float = 0.60
MAX_VOCAB_FEATURES: int = 256
TRAIN_EPOCHS: int = 60


def train_guardrail_model(
    model: DomainGuardrailNet,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    epochs: int = TRAIN_EPOCHS,
    batch_size: int = 16,
    lr: float = 4e-3,
) -> Tuple[List[float], List[float]]:
    """
    Entrena DomainGuardrailNet durante `epochs` épocas usando BCELoss y Adam.
    """
    model.to(device)
    dataset = TensorDataset(X_train, y_train.unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses: List[float] = []
    val_losses: List[float] = []

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.unsqueeze(1).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            probs = model(bx)
            loss = criterion(probs, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bx.size(0)
            total += bx.size(0)

        avg_train_loss = running_loss / total

        model.eval()
        with torch.no_grad():
            val_probs = model(X_val_dev)
            avg_val_loss = float(criterion(val_probs, y_val_dev).item())

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  [DomainGuardrailNet] Época {epoch:02d}/{epochs:02d} | "
                f"Train BCE Loss: {avg_train_loss:.6f} | Val BCE Loss: {avg_val_loss:.6f}"
            )

    return train_losses, val_losses


def evaluate_guardrail(
    model: DomainGuardrailNet,
    X_val: torch.Tensor,
    y_val: np.ndarray,
    device: torch.device,
    threshold: float = RECOMMENDED_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evalúa el modelo en el conjunto de validación con el umbral recomendado (`threshold = 0.60`),
    reportando Accuracy, Precision, Recall, F1 y Out-of-Domain Rejection Rate (Specificity).
    """
    model.eval()
    with torch.no_grad():
        val_probs = model(X_val.to(device)).cpu().numpy().ravel()

    preds = (val_probs >= threshold).astype(np.int32)
    y_true = y_val.astype(np.int32)

    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    # Out-of-Domain Rejection Rate = True Negatives / Total Class 0
    ood_mask = y_true == 0
    ood_rejection_rate = float(np.mean(preds[ood_mask] == 0)) if np.any(ood_mask) else 1.0

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "ood_rejection_rate": ood_rejection_rate,
        "val_probs": val_probs,
    }


def save_guardrail_plots(
    train_losses: List[float],
    val_losses: List[float],
    val_probs: np.ndarray,
    y_val: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """
    Genera y guarda en `artifacts/guardrail_loss_and_scores.png`:
    1. Curva de pérdida BCELoss (Train vs Validation) a lo largo de las 60 épocas.
    2. Distribución de puntajes de probabilidad en validación (Clase 0 vs Clase 1) con el umbral 0.60.
    """
    epochs_range = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=140)
    fig.suptitle(
        "Nano-AI - Clasificador Guardrail de Dominio Térmico (English TF-IDF + MLP)\n"
        "Inicialización Aleatoria (Kaiming/Xavier) | 60 Épocas | BCELoss + Adam",
        fontsize=12,
        fontweight="bold",
    )

    # Subplot 1: Curva de pérdida BCELoss
    axes[0].plot(epochs_range, train_losses, color="#1f77b4", linewidth=2.2, label="Train BCE Loss")
    axes[0].plot(epochs_range, val_losses, color="#ff7f0e", linewidth=2.2, linestyle="--", label="Val BCE Loss")
    axes[0].set_title("Convergencia de Entrenamiento (BCELoss)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Subplot 2: Distribución de probabilidades predichas
    ood_scores = val_probs[y_val == 0]
    ind_scores = val_probs[y_val == 1]
    bins = np.linspace(0.0, 1.0, 21)

    axes[1].hist(
        ood_scores,
        bins=bins,
        alpha=0.72,
        color="#d62728",
        edgecolor="black",
        label="Clase 0: Out-of-Domain / Injection",
    )
    axes[1].hist(
        ind_scores,
        bins=bins,
        alpha=0.72,
        color="#2ca02c",
        edgecolor="black",
        label="Clase 1: In-Domain Thermal Query",
    )
    axes[1].axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2.2,
        label=f"Umbral Guardrail = {threshold:.2f}",
    )
    axes[1].set_title("Distribución de Puntajes de Probabilidad (Validación)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Probabilidad de Dominio Térmico P(y = 1 | x)")
    axes[1].set_ylabel("Número de Frases")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[OK] Gráfica de pérdida y puntajes guardada en: {output_path}")


def export_and_verify_guardrail_onnx(
    model: DomainGuardrailNet,
    vocab_dict: Dict[str, Any],
    onnx_path: Path,
    threshold: float = RECOMMENDED_THRESHOLD,
) -> Dict[str, Any]:
    """
    Exporta `DomainGuardrailNet` a `artifacts/guardrail_skeleton.onnx` (`opset_version=17`,
    `dynamic_axes` en `batch_size`), valida con `onnx.checker.check_model` y ejecuta una prueba
    cruzada con `onnxruntime.InferenceSession` sobre 10 frases nuevas en inglés.
    """
    model_cpu = model.to("cpu")
    model_cpu.eval()

    input_dim = int(vocab_dict["input_dim"])
    dummy_input = torch.randn(4, input_dim, dtype=torch.float32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "export_params": True,
        "opset_version": 17,
        "do_constant_folding": True,
        "input_names": ["tfidf_features"],
        "output_names": ["in_domain_probability"],
        "dynamic_axes": {
            "tfidf_features": {0: "batch_size"},
            "in_domain_probability": {0: "batch_size"},
        },
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        torch.onnx.export(model_cpu, dummy_input, str(onnx_path), **export_kwargs)

    # 1. Validación estructural del grafo ONNX
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 2. Sesión ONNX Runtime y prueba con las 10 frases nuevas en inglés
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    phrases_only = [p for p, _ in UNSEEN_TEST_PHRASES_EN]
    expected_labels = [lbl for _, lbl in UNSEEN_TEST_PHRASES_EN]

    # Medición de tiempo de vectorización NumPy + inferencia ONNX Runtime
    t0 = time.perf_counter()
    X_test_np = vectorize_text_numpy(phrases_only, vocab_dict)
    ort_probs = ort_session.run(["in_domain_probability"], {"tfidf_features": X_test_np})[0].ravel()
    t1 = time.perf_counter()

    total_latency_ms = (t1 - t0) * 1000.0
    per_phrase_latency_ms = total_latency_ms / len(phrases_only)

    with torch.no_grad():
        torch_probs = model_cpu(torch.from_numpy(X_test_np)).numpy().ravel()

    max_abs_diff = float(np.max(np.abs(torch_probs - ort_probs)))
    if max_abs_diff >= 1e-6:
        raise AssertionError(
            f"La diferencia entre PyTorch y ONNX Runtime ({max_abs_diff:.2e}) excede el límite de 1e-6."
        )

    file_size_kb = onnx_path.stat().st_size / 1024.0
    print(
        f"[OK] ONNX exportado y validado: {onnx_path.name} ({file_size_kb:.2f} KB) | "
        f"onnx.checker: PASS | Max Diff PyTorch vs ORT: {max_abs_diff:.2e}"
    )
    print(
        f"[OK] Latencia de inferencia (Vectorización NumPy + ORT): "
        f"{total_latency_ms:.3f} ms total ({per_phrase_latency_ms:.4f} ms/frase)"
    )

    print("\n" + "=" * 100)
    print("  PRUEBA DE INFERENCIA EN ONNX RUNTIME - 10 FRASES NUEVAS EN INGLÉS (THRESHOLD = 0.60)")
    print("=" * 100)
    print(f"  {'#':<2} | {'Esperado':<10} | {'Pred ONNX':<10} | {'Prob PyTorch':<12} | {'Prob ONNX':<10} | {'Frase Evaluada'}")
    print("-" * 100)

    correct_count = 0
    test_results = []
    for idx, (phrase, exp_lbl) in enumerate(UNSEEN_TEST_PHRASES_EN, start=1):
        p_torch = float(torch_probs[idx - 1])
        p_onnx = float(ort_probs[idx - 1])
        pred_lbl = 1 if p_onnx >= threshold else 0
        is_ok = pred_lbl == exp_lbl
        if is_ok:
            correct_count += 1

        exp_str = "1 (VALID)" if exp_lbl == 1 else "0 (REJECT)"
        pred_str = "1 (VALID)" if pred_lbl == 1 else "0 (REJECT)"
        print(
            f"  {idx:02d} | {exp_str:<10} | {pred_str:<10} | {p_torch:<12.6f} | {p_onnx:<10.6f} | {phrase}"
        )
        test_results.append({
            "index": idx,
            "phrase": phrase,
            "expected": exp_lbl,
            "predicted": pred_lbl,
            "prob_pytorch": p_torch,
            "prob_onnx": p_onnx,
            "correct": is_ok,
        })

    print("-" * 100)
    print(
        f"  Exactitud en las 10 frases nuevas: {correct_count}/{len(UNSEEN_TEST_PHRASES_EN)} "
        f"({(correct_count / len(UNSEEN_TEST_PHRASES_EN)) * 100:.1f}%) | "
        f"Error Máx. PyTorch vs. ONNX: {max_abs_diff:.2e} (< 1e-6)"
    )
    print("=" * 100)

    return {
        "file_size_kb": file_size_kb,
        "max_abs_diff": max_abs_diff,
        "total_latency_ms": total_latency_ms,
        "per_phrase_latency_ms": per_phrase_latency_ms,
        "unseen_accuracy": correct_count / len(UNSEEN_TEST_PHRASES_EN),
        "test_results": test_results,
    }


def main() -> None:
    device = print_diagnostic_report()
    torch.manual_seed(42)
    np.random.seed(42)

    print("\n[Paso 1/5] Cargando banco balanceado de 120 frases en inglés (Split 80/20 estratificado)...")
    X_train_texts, X_val_texts, y_train, y_val = load_or_generate_guardrail_dataset(
        csv_path=None, test_size=0.20, random_state=42
    )
    print(
        f"  - Train set : {len(X_train_texts)} frases "
        f"(Clase 1 In-Domain: {int(y_train.sum())}, Clase 0 Out-of-Domain: {int((1 - y_train).sum())})"
    )
    print(
        f"  - Val set   : {len(X_val_texts)} frases "
        f"(Clase 1 In-Domain: {int(y_val.sum())}, Clase 0 Out-of-Domain: {int((1 - y_val).sum())})"
    )

    print("\n[Paso 2/5] Ajustando vectorizador TF-IDF híbrido únicamente sobre Train (sin data leakage)...")
    vocab_dict = fit_tfidf_vocab(
        train_texts=X_train_texts,
        train_labels=y_train,
        max_features=MAX_VOCAB_FEATURES,
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 4),
    )
    vocab_dict["recommended_threshold"] = RECOMMENDED_THRESHOLD
    save_tfidf_vocab(vocab_dict, GUARDRAIL_VOCAB_PATH)
    print(
        f"[OK] Vocabulario TF-IDF ({vocab_dict['input_dim']} características) guardado en: {GUARDRAIL_VOCAB_PATH}"
    )

    X_train_np = vectorize_text_numpy(X_train_texts, vocab_dict)
    X_val_np = vectorize_text_numpy(X_val_texts, vocab_dict)

    X_train_t = torch.from_numpy(X_train_np)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val_np)
    y_val_t = torch.from_numpy(y_val)

    print("\n[Paso 3/5] Instanciando y entrenando DomainGuardrailNet desde cero (60 épocas, BCELoss + Adam)...")
    model = DomainGuardrailNet(input_dim=int(vocab_dict["input_dim"]))
    summarize_guardrail_footprint(model, "DomainGuardrailNet (Guardrail Térmico)")

    train_losses, val_losses = train_guardrail_model(
        model=model,
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=X_val_t,
        y_val=y_val_t,
        device=device,
        epochs=TRAIN_EPOCHS,
        batch_size=16,
        lr=4e-3,
    )

    print("\n[Paso 4/5] Evaluando desempeño en Validación y generando gráficas...")
    val_metrics = evaluate_guardrail(
        model=model,
        X_val=X_val_t,
        y_val=y_val,
        device=device,
        threshold=RECOMMENDED_THRESHOLD,
    )
    print("  --------------------------------------------------------------------")
    print(f"  [Métricas de Validación Guardrail (Umbral recomendado = {RECOMMENDED_THRESHOLD:.2f})]")
    print(f"    - Accuracy  (Exactitud Global)          : {val_metrics['accuracy'] * 100:.2f}%")
    print(f"    - Precision (Consultas Térmicas Válidas): {val_metrics['precision'] * 100:.2f}%")
    print(f"    - Recall    (Consultas Térmicas Válidas): {val_metrics['recall'] * 100:.2f}%")
    print(f"    - F1-Score                              : {val_metrics['f1'] * 100:.2f}%")
    print(f"    - Out-of-Domain Rejection Rate (Clase 0): {val_metrics['ood_rejection_rate'] * 100:.2f}%")
    print("  --------------------------------------------------------------------")

    save_guardrail_plots(
        train_losses=train_losses,
        val_losses=val_losses,
        val_probs=val_metrics["val_probs"],
        y_val=y_val,
        threshold=RECOMMENDED_THRESHOLD,
        output_path=GUARDRAIL_PLOT_PATH,
    )

    print("\n[Paso 5/5] Exportando a ONNX (opset 17) y verificando 10 frases nuevas en ONNX Runtime...")
    export_and_verify_guardrail_onnx(
        model=model,
        vocab_dict=vocab_dict,
        onnx_path=GUARDRAIL_ONNX_PATH,
        threshold=RECOMMENDED_THRESHOLD,
    )


if __name__ == "__main__":
    main()
