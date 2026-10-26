"""
train_and_export_skeletons.py
Pipeline integral de la Semana 1 para el PC Training Node (Proyecto Nano-AI):
1. Diagnóstico de hardware y entorno (env_check).
2. Generación de series temporales térmicas sintéticas (DS18B20).
3. Normalización con StandardScaler y exportación de parámetros a JSON (artifacts/scaler_synthetic.json)
   para que la Raspberry Pi pueda aplicar exactamente el mismo preprocesamiento en C++/Python sin depender de pickle.
4. Entrenamiento corto de validación (30 épocas con MSELoss + Adam) desde pesos aleatorios (Kaiming/Xavier)
   verificando convergencia monotónica del Loss en ForecasterRegressor y AnomalyAutoencoder.
5. Generación de gráficas de convergencia en artifacts/loss_curves_skeleton.png.
6. Exportación de ambos modelos a formato ONNX (opset_version=17, dynamic_axes en batch_size):
   - artifacts/forecaster_skeleton.onnx
   - artifacts/autoencoder_skeleton.onnx
7. Validación estructural con onnx.checker.check_model y verificación de paridad numérica entre
   PyTorch y ONNX Runtime (onnxruntime.InferenceSession).

Autor: David Isaí Portillo López (R2 - Model Lead)
"""

from __future__ import annotations

import inspect
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica para ejecución limpia en cualquier entorno
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Asegurar que el directorio raíz del proyecto esté en sys.path si se ejecuta directamente
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_check import print_diagnostic_report
from src.models import AnomalyAutoencoder, ForecasterRegressor, summarize_model_footprint
from src.synthetic_data import generate_all_synthetic_datasets


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SCALER_JSON_PATH = ARTIFACTS_DIR / "scaler_synthetic.json"
LOSS_PLOT_PATH = ARTIFACTS_DIR / "loss_curves_skeleton.png"
FORECASTER_ONNX_PATH = ARTIFACTS_DIR / "forecaster_skeleton.onnx"
AUTOENCODER_ONNX_PATH = ARTIFACTS_DIR / "autoencoder_skeleton.onnx"


def fit_and_save_scalers(
    X_reg: np.ndarray,
    y_reg: np.ndarray,
    X_ae: np.ndarray,
    output_path: Path,
) -> Tuple[StandardScaler, StandardScaler, StandardScaler]:
    """
    Ajusta objetos StandardScaler para las entradas/salidas de ambos modelos y guarda
    sus medias y desviaciones estándar en formato JSON portable para el nodo Edge (Raspberry Pi).
    """
    scaler_x_reg = StandardScaler()
    scaler_y_reg = StandardScaler()
    scaler_x_ae = StandardScaler()

    scaler_x_reg.fit(X_reg)
    scaler_y_reg.fit(y_reg)
    scaler_x_ae.fit(X_ae)

    scaler_payload = {
        "metadata": {
            "project": "Nano-AI",
            "week": 1,
            "author": "David Isaí Portillo López (R2 - Model Lead)",
            "description": "Parámetros de estandarización z-score (x_norm = (x - mean) / scale) sobre datos sintéticos DS18B20",
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
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scaler_payload, f, indent=2, ensure_ascii=False)

    print(f"[OK] Escaladores StandardScaler guardados en: {output_path}")
    return scaler_x_reg, scaler_y_reg, scaler_x_ae


def train_forecaster_skeleton(
    model: nn.Module,
    X_norm: torch.Tensor,
    y_norm: torch.Tensor,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> List[float]:
    """Entrena el esqueleto ForecasterRegressor durante `epochs` épocas con MSELoss + Adam."""
    model.to(device)
    model.train()

    dataset = TensorDataset(X_norm, y_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses: List[float] = []
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total_samples = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

        avg_loss = running_loss / total_samples
        epoch_losses.append(avg_loss)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  [ForecasterRegressor] Época {epoch:02d}/{epochs:02d} | MSE Loss: {avg_loss:.6f}")

    return epoch_losses


def train_autoencoder_skeleton(
    model: nn.Module,
    X_norm: torch.Tensor,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 2e-3,
) -> List[float]:
    """Entrena el esqueleto AnomalyAutoencoder (reconstrucción X -> X) durante `epochs` épocas."""
    model.to(device)
    model.train()

    dataset = TensorDataset(X_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses: List[float] = []
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total_samples = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

        avg_loss = running_loss / total_samples
        epoch_losses.append(avg_loss)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  [AnomalyAutoencoder]  Época {epoch:02d}/{epochs:02d} | MSE Recon Loss: {avg_loss:.6f}")

    return epoch_losses


def save_loss_curves(
    reg_losses: List[float],
    ae_losses: List[float],
    output_path: Path,
) -> None:
    """Genera y guarda las gráficas de convergencia de entrenamiento de ambos modelos."""
    epochs_range = list(range(1, len(reg_losses) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    fig.suptitle(
        "Nano-AI (Semana 1) - Convergencia de Entrenamiento en Datos Sintéticos (DS18B20)\n"
        "Inicialización Aleatoria (Kaiming/Xavier) | Optimizador Adam | MSELoss",
        fontsize=12,
        fontweight="bold",
    )

    # Subplot 1: ForecasterRegressor
    axes[0].plot(epochs_range, reg_losses, color="#1f77b4", linewidth=2.2, marker="o", markersize=4, label="MSE Loss (Predicción +30 min)")
    axes[0].set_title("Track 4: ForecasterRegressor (20 -> 32 -> 16 -> 1)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("MSE Loss (Datos Normalizados)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Subplot 2: AnomalyAutoencoder
    axes[1].plot(epochs_range, ae_losses, color="#d62728", linewidth=2.2, marker="s", markersize=4, label="MSE Loss (Reconstrucción)")
    axes[1].set_title("Track 3: AnomalyAutoencoder (12 -> 8 -> 3 -> 8 -> 12)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MSE Loss de Reconstrucción")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[OK] Curvas de pérdida guardadas en: {output_path}")


def export_and_validate_onnx(
    model: nn.Module,
    input_dim: int,
    onnx_path: Path,
    input_name: str,
    output_name: str,
) -> Dict[str, float]:
    """
    Exporta un modelo PyTorch a ONNX con opset_version=17 y eje dinámico en batch_size,
    valida el grafo con onnx.checker.check_model y comprueba la paridad numérica contra ONNX Runtime.
    """
    model_cpu = model.to("cpu")
    model_cpu.eval()

    torch.manual_seed(123)
    dummy_input = torch.randn(4, input_dim, dtype=torch.float32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = {
        "export_params": True,
        "opset_version": 17,
        "do_constant_folding": True,
        "input_names": [input_name],
        "output_names": [output_name],
        "dynamic_axes": {
            input_name: {0: "batch_size"},
            output_name: {0: "batch_size"},
        },
    }
    # En PyTorch 2.6+, desactivar dynamo explícitamente permite usar dynamic_axes clásico sin warnings
    # y empaquetar todos los tensores dentro de un único archivo .onnx autocontenido para la Raspberry Pi.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        torch.onnx.export(model_cpu, dummy_input, str(onnx_path), **export_kwargs)

    # 1. Validación estructural del estándar ONNX
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 2. Prueba de inferencia y paridad numérica con ONNX Runtime (probando también batch dinámico N=7)
    test_input = torch.randn(7, input_dim, dtype=torch.float32)
    with torch.no_grad():
        torch_out = model_cpu(test_input).numpy()

    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {input_name: test_input.numpy()}
    ort_out = ort_session.run([output_name], ort_inputs)[0]

    max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-5, atol=1e-5)

    file_size_kb = onnx_path.stat().st_size / 1024.0
    print(
        f"[OK] ONNX exportado y validado: {onnx_path.name} ({file_size_kb:.2f} KB) | "
        f"onnx.checker: PASS | Max Diff PyTorch vs ORT: {max_abs_diff:.2e}"
    )

    return {
        "file_size_kb": file_size_kb,
        "max_abs_diff": max_abs_diff,
    }


def main() -> None:
    # 1. Diagnóstico del entorno
    device = print_diagnostic_report()

    # 2. Generación de datos sintéticos DS18B20
    print("\n[Paso 1/5] Generando series temporales sintéticas DS18B20...")
    torch.manual_seed(42)
    np.random.seed(42)
    datasets = generate_all_synthetic_datasets(n_samples=4096, seed=42)
    X_reg_raw = datasets["X_reg"].numpy()
    y_reg_raw = datasets["y_reg"].numpy()
    X_ae_raw = datasets["X_ae"].numpy()
    print(f"  Dimensiones generadas -> X_reg: {X_reg_raw.shape}, y_reg: {y_reg_raw.shape}, X_ae: {X_ae_raw.shape}")

    # 3. Normalización y persistencia en JSON
    print("\n[Paso 2/5] Ajustando StandardScaler y exportando metadatos...")
    scaler_x_reg, scaler_y_reg, scaler_x_ae = fit_and_save_scalers(
        X_reg_raw, y_reg_raw, X_ae_raw, SCALER_JSON_PATH
    )

    X_reg_norm = torch.from_numpy(scaler_x_reg.transform(X_reg_raw).astype(np.float32))
    y_reg_norm = torch.from_numpy(scaler_y_reg.transform(y_reg_raw).astype(np.float32))
    X_ae_norm = torch.from_numpy(scaler_x_ae.transform(X_ae_raw).astype(np.float32))

    # 4. Instanciación e inspección de modelos desde pesos aleatorios
    print("\n[Paso 3/5] Instanciando esqueletos con pesos aleatorios (Kaiming/Xavier)...")
    forecaster = ForecasterRegressor(input_dim=20, output_dim=1)
    autoencoder = AnomalyAutoencoder(input_dim=12, latent_dim=3)

    summarize_model_footprint(forecaster, "ForecasterRegressor (Track 4)")
    summarize_model_footprint(autoencoder, "AnomalyAutoencoder (Track 3)")

    # 5. Entrenamiento de validación (30 épocas)
    print("\n[Paso 4/5] Ejecutando entrenamiento de validación (30 épocas, Adam + MSELoss)...")
    reg_losses = train_forecaster_skeleton(forecaster, X_reg_norm, y_reg_norm, device=device, epochs=30)
    print("-" * 72)
    ae_losses = train_autoencoder_skeleton(autoencoder, X_ae_norm, device=device, epochs=30)

    if not (reg_losses[-1] < reg_losses[0] and ae_losses[-1] < ae_losses[0]):
        raise RuntimeError("La pérdida no disminuyó como se esperaba durante el entrenamiento de validación.")

    print(
        f"\n  Reducción de Loss verificada exitosamente:\n"
        f"    - ForecasterRegressor : {reg_losses[0]:.6f} -> {reg_losses[-1]:.6f} "
        f"(-{(1.0 - reg_losses[-1] / reg_losses[0]) * 100:.1f}%)\n"
        f"    - AnomalyAutoencoder  : {ae_losses[0]:.6f} -> {ae_losses[-1]:.6f} "
        f"(-{(1.0 - ae_losses[-1] / ae_losses[0]) * 100:.1f}%)"
    )

    save_loss_curves(reg_losses, ae_losses, LOSS_PLOT_PATH)

    # 6. Compilación y exportación a ONNX + Validación ORT
    print("\n[Paso 5/5] Exportando y validando modelos en formato ONNX (opset 17)...")
    export_and_validate_onnx(
        model=forecaster,
        input_dim=20,
        onnx_path=FORECASTER_ONNX_PATH,
        input_name="sensor_window_and_time",
        output_name="predicted_temp_plus_30m",
    )
    export_and_validate_onnx(
        model=autoencoder,
        input_dim=12,
        onnx_path=AUTOENCODER_ONNX_PATH,
        input_name="normal_window_features",
        output_name="reconstructed_window_features",
    )

    print("\n" + "=" * 72)
    print("  PIPELINE SEMANA 1 COMPLETADO EXITOSAMENTE - ARTEFACTOS GENERADOS:")
    print(f"    1. {SCALER_JSON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"    2. {LOSS_PLOT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"    3. {FORECASTER_ONNX_PATH.relative_to(PROJECT_ROOT)}")
    print(f"    4. {AUTOENCODER_ONNX_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
