"""
train_and_export_real.py
Pipeline de entrenamiento, validación, calibración de umbral y exportación ONNX con DATOS REALES
del sensor DS18B20 para el proyecto Nano-AI.

Flujo de trabajo:
1. Carga `data/train_solo_normales.csv` (100% libre de perturbaciones) para entrenar:
   - ForecasterRegressor (Track 4): Predicción de temperatura exacta a +30 min (+1800s).
   - AnomalyAutoencoder (Track 3): Reconstrucción de comportamiento térmico normal.
2. Carga `data/dataset_temperatura_unificado_limpio.csv` (que incluye `es_perturbacion_manual` 0 y 1)
   para calibrar automáticamente el umbral óptimo de anomalía (MSE Threshold) maximizando F1-Score
   y reportar Precision, Recall y F1 sobre las perturbaciones reales.
3. Guarda los parámetros de normalización y el umbral calibrado en `artifacts/scaler_real.json`.
4. Genera el tablero visual de resultados reales en `artifacts/loss_curves_real.png`.
5. Exporta y valida con `onnx.checker` y `onnxruntime` los modelos finales para la Raspberry Pi:
   - `artifacts/forecaster_real.onnx`
   - `artifacts/autoencoder_real.onnx`

Autor: David Isaí Portillo López (R2 - Model Lead)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_check import print_diagnostic_report
from src.models import AnomalyAutoencoder, ForecasterRegressor, summarize_model_footprint
from src.real_data import build_real_autoencoder_dataset, build_real_regressor_dataset
from src.train_and_export_skeletons import export_and_validate_onnx


DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "train_solo_normales.csv"
DEFAULT_EVAL_CSV = PROJECT_ROOT / "data" / "dataset_temperatura_unificado_limpio.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SCALER_REAL_JSON = ARTIFACTS_DIR / "scaler_real.json"
LOSS_PLOT_REAL_PATH = ARTIFACTS_DIR / "loss_curves_real.png"
FORECASTER_REAL_ONNX = ARTIFACTS_DIR / "forecaster_real.onnx"
AUTOENCODER_REAL_ONNX = ARTIFACTS_DIR / "autoencoder_real.onnx"


def train_regressor_real(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Tuple[List[float], List[float]]:
    """Entrena ForecasterRegressor con conjunto de entrenamiento/validación y restaura el mejor peso."""
    model.to(device)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bx.size(0)
            total += bx.size(0)

        t_loss = running_loss / total

        model.eval()
        with torch.no_grad():
            v_preds = model(X_val_dev)
            v_loss = float(criterion(v_preds, y_val_dev).item())

        scheduler.step(v_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  [ForecasterRegressor] Época {epoch:02d}/{epochs:02d} | "
                f"Train MSE: {t_loss:.5f} | Val MSE: {v_loss:.5f}"
            )

    model.load_state_dict(best_state)
    return train_losses, val_losses


def train_autoencoder_real(
    model: nn.Module,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    device: torch.device,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 2e-3,
) -> Tuple[List[float], List[float]]:
    """Entrena AnomalyAutoencoder exclusivamente con lecturas normales y restaura el mejor peso."""
    model.to(device)
    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    X_val_dev = X_val.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for (bx,) in train_loader:
            bx = bx.to(device)
            optimizer.zero_grad()

            # 1. Reconstrucción exacta sobre la variedad normal
            recon_clean = model(bx)
            loss_clean = criterion(recon_clean, bx)

            # 2. Regularización Denoising Autoencoder (100% no supervisada sobre datos normales):
            # Inyecta desplazamientos/picos sintéticos fuera de distribución (> 3 sigma) y fuerza
            # al decoder a proyectarlos de vuelta hacia la ventana normal `bx`, evitando que las
            # capas ReLU extrapolen linealmente temperaturas extremas.
            shift = (torch.randint(0, 2, (bx.size(0), 1), device=device).float() * 2.0 - 1.0) * (
                2.5 + 4.0 * torch.rand((bx.size(0), 1), device=device)
            )
            spike_noise = 0.5 * torch.randn_like(bx)
            bx_corrupted = bx + shift + spike_noise
            recon_corrupted = model(bx_corrupted)
            loss_denoise = criterion(recon_corrupted, bx)

            loss = loss_clean + 0.35 * loss_denoise
            loss.backward()
            optimizer.step()
            running_loss += loss_clean.item() * bx.size(0)
            total += bx.size(0)

        t_loss = running_loss / total

        model.eval()
        with torch.no_grad():
            v_recon = model(X_val_dev)
            v_loss = float(criterion(v_recon, X_val_dev).item())

        scheduler.step(v_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  [AnomalyAutoencoder]  Época {epoch:02d}/{epochs:02d} | "
                f"Train Recon MSE: {t_loss:.5f} | Val Recon MSE: {v_loss:.5f}"
            )

    model.load_state_dict(best_state)
    return train_losses, val_losses


def calibrate_anomaly_threshold(
    model: nn.Module,
    X_eval_norm: np.ndarray,
    y_eval_labels: np.ndarray,
    device: torch.device,
) -> Dict[str, float | np.ndarray]:
    """
    Calcula el error cuadrático medio (MSE) por muestra sobre el dataset unificado (con perturbaciones)
    y selecciona el umbral tau que maximiza el F1-Score entre lecturas normales (0) y perturbaciones (1).
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_eval_norm).to(device)
        recon = model(x_tensor).cpu().numpy()

    sample_mse = np.mean((X_eval_norm - recon) ** 2, axis=1)
    normal_mse = sample_mse[y_eval_labels == 0]

    # Búsqueda de umbral candidato entre percentiles de datos normales y el rango observado
    candidate_thresholds = np.unique(
        np.concatenate([
            np.percentile(normal_mse, np.linspace(90.0, 99.95, 80)),
            np.geomspace(max(float(np.median(normal_mse)), 1e-4), float(np.max(sample_mse)) * 0.8, 120),
        ])
    )

    best_thr = float(np.percentile(normal_mse, 99.0))
    best_f1 = -1.0

    if np.any(y_eval_labels == 1):
        for thr in candidate_thresholds:
            preds = (sample_mse > thr).astype(np.int32)
            score = float(f1_score(y_eval_labels, preds, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_thr = float(thr)

    final_preds = (sample_mse > best_thr).astype(np.int32)
    prec = float(precision_score(y_eval_labels, final_preds, zero_division=0))
    rec = float(recall_score(y_eval_labels, final_preds, zero_division=0))
    f1 = float(f1_score(y_eval_labels, final_preds, zero_division=0))

    return {
        "threshold": best_thr,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "normal_mse_p99": float(np.percentile(normal_mse, 99.0)),
        "sample_mse": sample_mse,
    }


def save_real_diagnostics_plot(
    reg_train_losses: List[float],
    reg_val_losses: List[float],
    y_val_true_c: np.ndarray,
    y_val_pred_c: np.ndarray,
    ae_train_losses: List[float],
    ae_val_losses: List[float],
    sample_mse: np.ndarray,
    y_eval_labels: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Genera un panel de 4 gráficas con el desempeño de ambos modelos sobre datos reales."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=140)
    fig.suptitle(
        "Nano-AI - Entrenamiento y Validación con Datos Reales del Sensor DS18B20\n"
        "ForecasterRegressor (+30 min) & AnomalyAutoencoder (Detección de Perturbaciones)",
        fontsize=12.5,
        fontweight="bold",
    )

    # 1. Curva de pérdida Regresor
    ep_reg = range(1, len(reg_train_losses) + 1)
    axes[0, 0].plot(ep_reg, reg_train_losses, label="Train MSE", color="#1f77b4", linewidth=2)
    axes[0, 0].plot(ep_reg, reg_val_losses, label="Val MSE", color="#ff7f0e", linestyle="--", linewidth=2)
    axes[0, 0].set_title("Track 4: Convergencia ForecasterRegressor", fontweight="bold")
    axes[0, 0].set_xlabel("Época")
    axes[0, 0].set_ylabel("MSE Loss (Normalizado)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    # 2. Predicción vs Real en °C (Muestra de validación)
    n_plot = min(250, len(y_val_true_c))
    axes[0, 1].plot(range(n_plot), y_val_true_c[:n_plot], label="Real DS18B20 (+30 min)", color="#2ca02c", linewidth=1.8)
    axes[0, 1].plot(range(n_plot), y_val_pred_c[:n_plot], label="Predicción MLP (+30 min)", color="#1f77b4", linestyle="--", linewidth=1.8)
    axes[0, 1].set_title("Track 4: Temperatura Real vs. Predicha (+30 min) [°C]", fontweight="bold")
    axes[0, 1].set_xlabel("Muestras de Validación")
    axes[0, 1].set_ylabel("Temperatura (°C)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    # 3. Curva de pérdida Autoencoder
    ep_ae = range(1, len(ae_train_losses) + 1)
    axes[1, 0].plot(ep_ae, ae_train_losses, label="Train Recon MSE", color="#d62728", linewidth=2)
    axes[1, 0].plot(ep_ae, ae_val_losses, label="Val Recon MSE", color="#9467bd", linestyle="--", linewidth=2)
    axes[1, 0].set_title("Track 3: Convergencia AnomalyAutoencoder", fontweight="bold")
    axes[1, 0].set_xlabel("Época")
    axes[1, 0].set_ylabel("MSE de Reconstrucción")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()

    # 4. Separación de MSE Normal vs Perturbación Manual
    normal_err = np.clip(sample_mse[y_eval_labels == 0], 1e-5, None)
    anom_err = np.clip(sample_mse[y_eval_labels == 1], 1e-5, None)
    bins = np.logspace(np.log10(min(normal_err.min(), 1e-4)), np.log10(max(sample_mse.max(), 10.0)), 50)
    axes[1, 1].hist(normal_err, bins=bins, alpha=0.65, color="#2ca02c", label="Normal (es_perturbacion=0)")
    if len(anom_err) > 0:
        axes[1, 1].hist(anom_err, bins=bins, alpha=0.65, color="#d62728", label="Perturbación (es_perturbacion=1)")
    axes[1, 1].axvline(threshold, color="black", linestyle="--", linewidth=2.2, label=f"Umbral τ = {threshold:.4f}")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("Track 3: Distribución de Error MSE (Escala Log)", fontweight="bold")
    axes[1, 1].set_xlabel("MSE de Reconstrucción (log)")
    axes[1, 1].set_ylabel("Frecuencia de Ventanas")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[OK] Gráficas de entrenamiento real guardadas en: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y exporta los modelos Nano-AI usando los CSV reales del sensor DS18B20."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help="Ruta al CSV sin perturbaciones para entrenamiento (default: data/train_solo_normales.csv)",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=DEFAULT_EVAL_CSV,
        help="Ruta al CSV unificado con perturbaciones para evaluar el Autoencoder (default: data/dataset_temperatura_unificado_limpio.csv)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Número de épocas de entrenamiento (default: 200)")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamaño de lote (default: 64)")
    args = parser.parse_args()

    device = print_diagnostic_report()
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"\n[Paso 1/6] Cargando datos reales de entrenamiento desde: {args.train_csv}")
    X_reg_raw, y_reg_raw = build_real_regressor_dataset(args.train_csv)
    X_ae_raw, _ = build_real_autoencoder_dataset(args.train_csv, only_normal=True)

    eval_csv_path = args.eval_csv if args.eval_csv.exists() else args.train_csv
    X_ae_eval_raw, y_ae_eval_labels = build_real_autoencoder_dataset(eval_csv_path, only_normal=False)

    print(f"  - Ventanas válidas Regresor (+30 min)  : {X_reg_raw.shape[0]:,} muestras de forma {X_reg_raw.shape[1]}")
    print(f"  - Ventanas válidas Autoencoder (Normal): {X_ae_raw.shape[0]:,} muestras de forma {X_ae_raw.shape[1]}")
    print(
        f"  - Ventanas evaluación Autoencoder      : {X_ae_eval_raw.shape[0]:,} "
        f"(Normales: {int(np.sum(y_ae_eval_labels == 0)):,}, Perturbaciones: {int(np.sum(y_ae_eval_labels == 1)):,})"
    )

    # División Entrenamiento (80%) / Validación (20%)
    X_reg_tr, X_reg_va, y_reg_tr, y_reg_va = train_test_split(
        X_reg_raw, y_reg_raw, test_size=0.20, random_state=42, shuffle=True
    )
    X_ae_tr, X_ae_va = train_test_split(
        X_ae_raw, test_size=0.20, random_state=42, shuffle=True
    )

    print("\n[Paso 2/6] Ajustando StandardScaler sobre partición de entrenamiento real...")
    scaler_x_reg = StandardScaler().fit(X_reg_tr)
    scaler_y_reg = StandardScaler().fit(y_reg_tr)
    scaler_x_ae = StandardScaler().fit(X_ae_tr)

    X_reg_tr_t = torch.from_numpy(scaler_x_reg.transform(X_reg_tr).astype(np.float32))
    y_reg_tr_t = torch.from_numpy(scaler_y_reg.transform(y_reg_tr).astype(np.float32))
    X_reg_va_t = torch.from_numpy(scaler_x_reg.transform(X_reg_va).astype(np.float32))
    y_reg_va_t = torch.from_numpy(scaler_y_reg.transform(y_reg_va).astype(np.float32))

    X_ae_tr_t = torch.from_numpy(scaler_x_ae.transform(X_ae_tr).astype(np.float32))
    X_ae_va_t = torch.from_numpy(scaler_x_ae.transform(X_ae_va).astype(np.float32))
    X_ae_eval_norm = scaler_x_ae.transform(X_ae_eval_raw).astype(np.float32)

    print("\n[Paso 3/6] Instanciando modelos desde pesos aleatorios (Kaiming/Xavier)...")
    forecaster = ForecasterRegressor(input_dim=20, output_dim=1)
    autoencoder = AnomalyAutoencoder(input_dim=12, latent_dim=3)
    summarize_model_footprint(forecaster, "ForecasterRegressor (Track 4 - Real Data)")
    summarize_model_footprint(autoencoder, "AnomalyAutoencoder (Track 3 - Real Data)")

    print(f"\n[Paso 4/6] Entrenando modelos con datos reales ({args.epochs} épocas)...")
    reg_tr_losses, reg_va_losses = train_regressor_real(
        forecaster, X_reg_tr_t, y_reg_tr_t, X_reg_va_t, y_reg_va_t,
        device=device, epochs=args.epochs, batch_size=args.batch_size
    )
    print("-" * 72)
    ae_tr_losses, ae_va_losses = train_autoencoder_real(
        autoencoder, X_ae_tr_t, X_ae_va_t,
        device=device, epochs=args.epochs, batch_size=args.batch_size
    )

    # Evaluación en grados Celsius (°C) para el Regresor
    forecaster.eval()
    with torch.no_grad():
        preds_va_norm = forecaster(X_reg_va_t.to(device)).cpu().numpy()
    y_val_pred_c = scaler_y_reg.inverse_transform(preds_va_norm).ravel()
    y_val_true_c = y_reg_va.ravel()

    mae_c = float(np.mean(np.abs(y_val_true_c - y_val_pred_c)))
    rmse_c = float(np.sqrt(np.mean((y_val_true_c - y_val_pred_c) ** 2)))
    r2_val = float(r2_score(y_val_true_c, y_val_pred_c))

    # Calibración de umbral y métricas para el Autoencoder
    print("\n[Paso 5/6] Evaluando métricas reales y calibrando umbral de anomalías...")
    ae_metrics = calibrate_anomaly_threshold(autoencoder, X_ae_eval_norm, y_ae_eval_labels, device=device)

    print("  --------------------------------------------------------------------")
    print(f"  [Métricas ForecasterRegressor (+30 min en Validación)]")
    print(f"    - MAE  (Error Absoluto Medio)  : {mae_c:.4f} °C")
    print(f"    - RMSE (Error Cuadrático Medio): {rmse_c:.4f} °C")
    print(f"    - R²   (Varianza Explicada)    : {r2_val:.4f}")
    print("  --------------------------------------------------------------------")
    print(f"  [Métricas AnomalyAutoencoder (sobre {eval_csv_path.name})]")
    print(f"    - Umbral Óptimo MSE (tau)      : {ae_metrics['threshold']:.6f}")
    print(f"    - Precision (Perturbaciones)   : {ae_metrics['precision'] * 100:.2f}%")
    print(f"    - Recall    (Perturbaciones)   : {ae_metrics['recall'] * 100:.2f}%")
    print(f"    - F1-Score  (Perturbaciones)   : {ae_metrics['f1'] * 100:.2f}%")
    print("  --------------------------------------------------------------------")

    # Guardar escalador real + umbral en JSON
    scaler_payload = {
        "metadata": {
            "project": "Nano-AI",
            "dataset_train": args.train_csv.name,
            "dataset_eval": eval_csv_path.name,
            "author": "David Isaí Portillo López (R2 - Model Lead)",
            "regressor_val_mae_celsius": round(mae_c, 5),
            "regressor_val_rmse_celsius": round(rmse_c, 5),
            "regressor_val_r2": round(r2_val, 5),
            "autoencoder_f1_score": round(float(ae_metrics["f1"]), 5),
        },
        "forecaster_regressor": {
            "input_features": [
                *[f"temp_t_minus_{17 - i}" for i in range(18)],
                "hour_sin",
                "hour_cos",
            ],
            "x_mean": scaler_x_reg.mean_.tolist(),
            "x_scale": scaler_x_reg.scale_.tolist(),
            "y_mean": scaler_y_reg.mean_.tolist(),
            "y_scale": scaler_y_reg.scale_.tolist(),
        },
        "anomaly_autoencoder": {
            "input_features": [
                *[f"temp_t_minus_{9 - i}" for i in range(10)],
                "moving_avg_10",
                "delta_t",
            ],
            "x_mean": scaler_x_ae.mean_.tolist(),
            "x_scale": scaler_x_ae.scale_.tolist(),
            "anomaly_mse_threshold": float(ae_metrics["threshold"]),
            "normal_mse_p99": float(ae_metrics["normal_mse_p99"]),
        },
    }
    SCALER_REAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_REAL_JSON, "w", encoding="utf-8") as f:
        json.dump(scaler_payload, f, indent=2, ensure_ascii=False)
    print(f"[OK] Escaladores reales y umbral guardados en: {SCALER_REAL_JSON}")

    save_real_diagnostics_plot(
        reg_tr_losses,
        reg_va_losses,
        y_val_true_c,
        y_val_pred_c,
        ae_tr_losses,
        ae_va_losses,
        ae_metrics["sample_mse"],
        y_ae_eval_labels,
        float(ae_metrics["threshold"]),
        LOSS_PLOT_REAL_PATH,
    )

    print("\n[Paso 6/6] Exportando y validando modelos reales a ONNX (opset 17)...")
    export_and_validate_onnx(
        model=forecaster,
        input_dim=20,
        onnx_path=FORECASTER_REAL_ONNX,
        input_name="sensor_window_and_time",
        output_name="predicted_temp_plus_30m",
    )
    export_and_validate_onnx(
        model=autoencoder,
        input_dim=12,
        onnx_path=AUTOENCODER_REAL_ONNX,
        input_name="normal_window_features",
        output_name="reconstructed_window_features",
    )

    print("\n" + "=" * 72)
    print("  ENTRENAMIENTO CON DATOS REALES COMPLETADO EXITOSAMENTE:")
    print(f"    1. {SCALER_REAL_JSON.relative_to(PROJECT_ROOT)}")
    print(f"    2. {LOSS_PLOT_REAL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"    3. {FORECASTER_REAL_ONNX.relative_to(PROJECT_ROOT)}")
    print(f"    4. {AUTOENCODER_REAL_ONNX.relative_to(PROJECT_ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
