"""
synthetic_data.py
Generador de series temporales térmicas sintéticas realistas emulando el sensor DS18B20
para el entrenamiento y validación inicial de esqueletos del proyecto Nano-AI (Semana 1).

Especificaciones:
- Curva térmica diaria base: onda senoidal entre 20.0 °C y 28.0 °C (media 24.0 °C, amplitud 4.0 °C)
  con período de 24 horas + ruido gaussiano suave propio del DS18B20 (resolución 12-bit ~0.0625 °C).
- Frecuencia de muestreo simulada: 1 lectura cada 10 segundos (360 lecturas/hora, 8640 lecturas/día).
- Dataset Regresor (Track 4):
    X_reg: (N, 20) -> [18 lecturas (3 min), hour_sin, hour_cos]
    y_reg: (N, 1)  -> [temperatura exacta a +30 min (+180 pasos de 10s)]
- Dataset Autoencoder (Track 3):
    X_ae:  (N, 12) -> [10 lecturas recientes, media_movil_10, delta_t]

Autor: David Isaí Portillo López (R2 - Model Lead)
"""

from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
import torch


SAMPLING_INTERVAL_SEC: int = 10
SECONDS_PER_HOUR: float = 3600.0
SECONDS_PER_DAY: float = 86400.0

REG_HISTORY_STEPS: int = 18       # 18 lecturas * 10s = 3 minutos
REG_HORIZON_STEPS: int = 180      # +30 minutos * 60s / 10s = 180 pasos hacia adelante
AE_HISTORY_STEPS: int = 10        # 10 lecturas recientes


def generate_ds18b20_series(
    total_steps: int,
    temp_min: float = 20.0,
    temp_max: float = 28.0,
    noise_std: float = 0.08,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera una serie temporal continua de temperatura ambiente simulando lecturas de un sensor DS18B20.

    Retorna:
        timestamps_sec: Vector de tiempo en segundos (total_steps,)
        clean_temp: Temperatura ambiente ideal sin ruido (total_steps,)
        noisy_temp: Lecturas del sensor DS18B20 con ruido gaussiano suave (total_steps,)
    """
    rng = np.random.default_rng(seed)
    timestamps_sec = np.arange(total_steps, dtype=np.float64) * SAMPLING_INTERVAL_SEC

    temp_mean = 0.5 * (temp_min + temp_max)  # 24.0 °C
    temp_amp = 0.5 * (temp_max - temp_min)   # 4.0 °C

    # Fase ajustada para que el mínimo ocurra en la madrugada (~05:00h) y el pico en la tarde (~17:00h)
    daily_phase = 2.0 * np.pi * (timestamps_sec / SECONDS_PER_DAY - 5.0 / 24.0)
    # Armónico secundario suave para dar realismo a la transición térmica diurna/nocturna
    harmonic = 0.25 * np.sin(2.0 * daily_phase)

    clean_temp = temp_mean - temp_amp * np.cos(daily_phase) + harmonic
    sensor_noise = rng.normal(loc=0.0, scale=noise_std, size=total_steps)
    noisy_temp = (clean_temp + sensor_noise).astype(np.float32)

    return timestamps_sec.astype(np.float32), clean_temp.astype(np.float32), noisy_temp


def build_regressor_dataset(
    n_samples: int = 4096,
    stride_steps: int = 3,
    seed: int = 42,
    as_tensors: bool = True,
) -> Tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
    """
    Construye los tensores para ForecasterRegressor (Track 4):
    - X_reg: (N, 20) -> 18 lecturas históricas (cada 10s = 3 min) + hour_sin + hour_cos
    - y_reg: (N, 1)  -> Temperatura ambiente a +30 minutos (+180 pasos de 10s)
    """
    required_steps = REG_HISTORY_STEPS + (n_samples - 1) * stride_steps + REG_HORIZON_STEPS + 1
    timestamps_sec, _, noisy_temp = generate_ds18b20_series(total_steps=required_steps, seed=seed)

    X_reg = np.zeros((n_samples, 20), dtype=np.float32)
    y_reg = np.zeros((n_samples, 1), dtype=np.float32)

    for i in range(n_samples):
        start_idx = i * stride_steps
        end_idx = start_idx + REG_HISTORY_STEPS  # exclusivo; última lectura de la ventana es end_idx - 1
        current_idx = end_idx - 1
        target_idx = current_idx + REG_HORIZON_STEPS

        # 18 lecturas de la ventana de 3 minutos
        window_18 = noisy_temp[start_idx:end_idx]

        # Hora del día en el instante actual (0 a 24 h) codificada cíclicamente
        hour_of_day = (timestamps_sec[current_idx] % SECONDS_PER_DAY) / SECONDS_PER_HOUR
        angle = 2.0 * np.pi * (hour_of_day / 24.0)
        hour_sin = np.sin(angle, dtype=np.float32)
        hour_cos = np.cos(angle, dtype=np.float32)

        X_reg[i, :18] = window_18
        X_reg[i, 18] = hour_sin
        X_reg[i, 19] = hour_cos

        # Objetivo: temperatura dentro de +30 minutos
        y_reg[i, 0] = noisy_temp[target_idx]

    if as_tensors:
        return torch.from_numpy(X_reg), torch.from_numpy(y_reg)
    return X_reg, y_reg


def build_autoencoder_dataset(
    n_samples: int = 4096,
    stride_steps: int = 3,
    seed: int = 101,
    as_tensors: bool = True,
) -> torch.Tensor | np.ndarray:
    """
    Construye el tensor de comportamiento normal para AnomalyAutoencoder (Track 3):
    - X_ae: (N, 12) -> 10 lecturas recientes + media móvil de la ventana + delta_t (T_t - T_{t-1})
    """
    required_steps = AE_HISTORY_STEPS + (n_samples - 1) * stride_steps + 1
    _, _, noisy_temp = generate_ds18b20_series(total_steps=required_steps, seed=seed)

    X_ae = np.zeros((n_samples, 12), dtype=np.float32)

    for i in range(n_samples):
        start_idx = i * stride_steps
        end_idx = start_idx + AE_HISTORY_STEPS
        window_10 = noisy_temp[start_idx:end_idx]

        moving_avg = float(np.mean(window_10))
        delta_t = float(window_10[-1] - window_10[-2])

        X_ae[i, :10] = window_10
        X_ae[i, 10] = moving_avg
        X_ae[i, 11] = delta_t

    if as_tensors:
        return torch.from_numpy(X_ae)
    return X_ae


def generate_all_synthetic_datasets(
    n_samples: int = 4096,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Genera y devuelve en un diccionario todos los tensores sintéticos requeridos en la Semana 1:
    - 'X_reg': (N, 20)
    - 'y_reg': (N, 1)
    - 'X_ae':  (N, 12)
    """
    X_reg, y_reg = build_regressor_dataset(n_samples=n_samples, seed=seed, as_tensors=True)
    X_ae = build_autoencoder_dataset(n_samples=n_samples, seed=seed + 1, as_tensors=True)
    return {
        "X_reg": X_reg,
        "y_reg": y_reg,
        "X_ae": X_ae,
    }


if __name__ == "__main__":
    datasets = generate_all_synthetic_datasets(n_samples=2048)
    X_r, y_r, X_a = datasets["X_reg"], datasets["y_reg"], datasets["X_ae"]
    print("=== Verificación de Datos Sintéticos (DS18B20) ===")
    print(f"X_reg shape : {tuple(X_r.shape)} | dtype: {X_r.dtype} | T min/max: [{X_r[:, :18].min():.2f}, {X_r[:, :18].max():.2f}] °C")
    print(f"y_reg shape : {tuple(y_r.shape)} | dtype: {y_r.dtype} | y min/max: [{y_r.min():.2f}, {y_r.max():.2f}] °C")
    print(f"X_ae  shape : {tuple(X_a.shape)} | dtype: {X_a.dtype} | media_movil prom: {X_a[:, 10].mean():.2f} °C | delta_t std: {X_a[:, 11].std():.4f} °C")
