"""
real_data.py
Cargador y procesador de series temporales reales del sensor DS18B20 desde archivos CSV.

Columnas esperadas en el CSV:
    timestamp_iso,epoch_segundos,fecha,hora_decimal,lectura_serial_cruda,temp_c,es_perturbacion_manual

Características clave:
- Respeta la continuidad temporal usando `epoch_segundos`: descarta cualquier ventana que cruce
  cortes de sesión o huecos temporales (ej. donde se filtraron perturbaciones).
- Para ForecasterRegressor (Track 4):
    X_reg: (N, 20) -> [18 lecturas consecutivas limpias, hour_sin, hour_cos]
    y_reg: (N, 1)  -> [temp_c exactamente a +1800 segundos (+30 min) dentro de una tolerancia de ±25s]
- Para AnomalyAutoencoder (Track 3):
    X_ae:  (N, 12) -> [10 lecturas consecutivas, media_movil_10, delta_t (T_t - T_{t-1})]
    También genera pares (X_ae_eval, y_anom_label) a partir del CSV unificado con perturbaciones
    para calibrar el umbral de reconstrucción MSE y medir Precision/Recall/F1.

Autor: David Isaí Portillo López (R2 - Model Lead)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


REQUIRED_COLUMNS = {
    "timestamp_iso",
    "epoch_segundos",
    "fecha",
    "hora_decimal",
    "lectura_serial_cruda",
    "temp_c",
    "es_perturbacion_manual",
}

REG_WINDOW_SIZE: int = 18
HORIZON_SECONDS: float = 1800.0  # +30 minutos en segundos
HORIZON_TOLERANCE_SEC: float = 25.0
MAX_STEP_GAP_SEC: float = 25.0   # Máximo salto permitido entre muestras consecutivas (~5s nominal)
AE_WINDOW_SIZE: int = 10


def load_raw_csv(csv_path: Path | str) -> Dict[str, np.ndarray]:
    """
    Lee un archivo CSV del sensor DS18B20, valida sus columnas y devuelve arreglos NumPy ordenados
    cronológicamente por `epoch_segundos`.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV en la ruta: {path}")

    epochs_list = []
    hours_list = []
    temps_list = []
    pert_list = []

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"El archivo CSV está vacío o no tiene encabezados: {path}")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Al archivo {path.name} le faltan las columnas requeridas: {sorted(missing)}"
            )

        for row in reader:
            epochs_list.append(float(row["epoch_segundos"]))
            hours_list.append(float(row["hora_decimal"]))
            temps_list.append(float(row["temp_c"]))
            pert_list.append(int(float(row["es_perturbacion_manual"])))

    epochs = np.asarray(epochs_list, dtype=np.float64)
    hours = np.asarray(hours_list, dtype=np.float32)
    temps = np.asarray(temps_list, dtype=np.float32)
    pert = np.asarray(pert_list, dtype=np.int32)

    # Asegurar orden cronológico estricto por epoch_segundos
    order = np.argsort(epochs, kind="mergesort")
    return {
        "epoch_segundos": epochs[order],
        "hora_decimal": hours[order],
        "temp_c": temps[order],
        "es_perturbacion_manual": pert[order],
    }


def build_real_regressor_dataset(
    csv_path: Path | str,
    horizon_sec: float = HORIZON_SECONDS,
    tolerance_sec: float = HORIZON_TOLERANCE_SEC,
    max_step_gap_sec: float = MAX_STEP_GAP_SEC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye (X_reg, y_reg) a partir del CSV de datos normales:
    - Filtra únicamente muestras donde toda la ventana de 18 lecturas y el objetivo a +30 min
      sean normales (`es_perturbacion_manual == 0`).
    - Verifica que no existan cortes de captura dentro de la ventana de entrada (`diff <= max_step_gap_sec`).
    - Busca el instante `t_actual + 1800s` mediante búsqueda binaria sobre `epoch_segundos`.
    """
    data = load_raw_csv(csv_path)
    epochs = data["epoch_segundos"]
    hours = data["hora_decimal"]
    temps = data["temp_c"]
    pert = data["es_perturbacion_manual"]

    n_rows = len(epochs)
    X_list = []
    y_list = []

    for current_idx in range(REG_WINDOW_SIZE - 1, n_rows):
        start_idx = current_idx - REG_WINDOW_SIZE + 1
        end_idx = current_idx + 1

        # 1. Descartar si hay alguna perturbación manual en la ventana de entrada
        if np.any(pert[start_idx:end_idx] != 0):
            continue

        # 2. Verificar continuidad temporal estricta dentro de las 18 lecturas
        window_epochs = epochs[start_idx:end_idx]
        if np.max(np.diff(window_epochs)) > max_step_gap_sec:
            continue

        # 3. Buscar la lectura correspondiente a +30 minutos (+1800 segundos)
        target_time = window_epochs[-1] + horizon_sec
        target_idx = int(np.searchsorted(epochs, target_time, side="left"))

        # Evaluar el candidato más cercano entre target_idx y target_idx - 1
        best_idx = -1
        best_diff = float("inf")
        for cand in (target_idx - 1, target_idx):
            if 0 <= cand < n_rows:
                diff = abs(epochs[cand] - target_time)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = cand

        if best_idx == -1 or best_diff > tolerance_sec:
            continue

        # El valor objetivo a +30 min también debe ser una lectura normal (sin perturbación)
        if pert[best_idx] != 0:
            continue

        window_temps = temps[start_idx:end_idx]
        hour_val = float(hours[current_idx])
        angle = 2.0 * np.pi * (hour_val / 24.0)
        hour_sin = np.sin(angle, dtype=np.float32)
        hour_cos = np.cos(angle, dtype=np.float32)

        features = np.empty(20, dtype=np.float32)
        features[:18] = window_temps
        features[18] = hour_sin
        features[19] = hour_cos

        X_list.append(features)
        y_list.append([temps[best_idx]])

    if not X_list:
        raise RuntimeError(
            f"No se pudieron construir ventanas válidas para el Regresor desde {csv_path}."
        )

    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def build_real_autoencoder_dataset(
    csv_path: Path | str,
    only_normal: bool = True,
    max_step_gap_sec: float = MAX_STEP_GAP_SEC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye ventanas (X_ae, y_label) de dimensión (N, 12) para el AnomalyAutoencoder:
    - 10 lecturas consecutivas + media móvil de las 10 lecturas + delta_t (T_t - T_{t-1}).
    - Si `only_normal=True`, devuelve únicamente ventanas donde las 10 lecturas tienen
      `es_perturbacion_manual == 0` (para entrenamiento limpio).
    - Si `only_normal=False`, devuelve todas las ventanas continuas junto con la etiqueta binaria
      `y_label` (1 si la ventana contiene perturbación manual, 0 si es completamente normal).
    """
    data = load_raw_csv(csv_path)
    epochs = data["epoch_segundos"]
    temps = data["temp_c"]
    pert = data["es_perturbacion_manual"]

    n_rows = len(epochs)
    X_list = []
    labels_list = []

    for current_idx in range(AE_WINDOW_SIZE - 1, n_rows):
        start_idx = current_idx - AE_WINDOW_SIZE + 1
        end_idx = current_idx + 1

        window_epochs = epochs[start_idx:end_idx]
        if np.max(np.diff(window_epochs)) > max_step_gap_sec:
            continue

        window_pert = pert[start_idx:end_idx]
        is_anomaly = int(np.any(window_pert != 0))

        if only_normal and is_anomaly == 1:
            continue

        window_temps = temps[start_idx:end_idx]
        moving_avg = float(np.mean(window_temps))
        delta_t = float(window_temps[-1] - window_temps[-2])

        features = np.empty(12, dtype=np.float32)
        features[:10] = window_temps
        features[10] = moving_avg
        features[11] = delta_t

        X_list.append(features)
        labels_list.append(is_anomaly)

    if not X_list:
        raise RuntimeError(
            f"No se pudieron construir ventanas válidas para el Autoencoder desde {csv_path}."
        )

    return np.asarray(X_list, dtype=np.float32), np.asarray(labels_list, dtype=np.int32)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    normal_csv = project_root / "data" / "train_solo_normales.csv"
    unified_csv = project_root / "data" / "dataset_temperatura_unificado_limpio.csv"

    X_reg, y_reg = build_real_regressor_dataset(normal_csv)
    X_ae_train, _ = build_real_autoencoder_dataset(normal_csv, only_normal=True)
    X_ae_eval, y_ae_eval = build_real_autoencoder_dataset(unified_csv, only_normal=False)

    print("=== Resumen de Datasets Reales (Sensor DS18B20) ===")
    print(f"1. ForecasterRegressor (desde {normal_csv.name}):")
    print(f"   - X_reg: {X_reg.shape} | y_reg: {y_reg.shape}")
    print(f"   - Rango térmico entrada: [{X_reg[:, :18].min():.2f}, {X_reg[:, :18].max():.2f}] °C")
    print(f"   - Rango térmico +30 min: [{y_reg.min():.2f}, {y_reg.max():.2f}] °C")
    print(f"2. AnomalyAutoencoder Entrenamiento (desde {normal_csv.name}):")
    print(f"   - X_ae_train (100% normales): {X_ae_train.shape}")
    print(f"3. AnomalyAutoencoder Evaluación (desde {unified_csv.name}):")
    print(
        f"   - X_ae_eval: {X_ae_eval.shape} | Normales: {int(np.sum(y_ae_eval == 0))} | "
        f"Con Perturbación: {int(np.sum(y_ae_eval == 1))}"
    )
