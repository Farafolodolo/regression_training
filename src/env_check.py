"""
env_check.py
Diagnóstico del entorno de entrenamiento (PC Training Node) para el proyecto Nano-AI.
Verifica versiones de librerías críticas (PyTorch, ONNX, ONNX Runtime, NumPy, Scikit-Learn)
y detecta automáticamente el backend de aceleración por hardware (CUDA, ROCm/HIP, MPS o CPU).

Autor: David Isaí Portillo López (R2 - Model Lead)
"""

from __future__ import annotations

import platform
import sys
from typing import Dict, Any

import numpy as np
import onnx
import onnxruntime as ort
import onnxscript
import sklearn
import torch


def get_compute_device() -> torch.device:
    """
    Detecta de forma automática el mejor dispositivo de cómputo disponible:
    CUDA (NVIDIA) / ROCm (AMD vía torch.version.hip) -> MPS (Apple Silicon) -> CPU fallback limpio.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_environment_info() -> Dict[str, Any]:
    """Recopila metadatos de hardware, sistema operativo y versiones del stack ML/ONNX."""
    device = get_compute_device()

    # Identificación específica entre NVIDIA CUDA y AMD ROCm (ambos exponen torch.cuda en PyTorch)
    accel_backend = "CPU (Fallback limpio)"
    device_name = platform.processor() or "CPU Genérico"

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
        if getattr(torch.version, "hip", None) is not None:
            accel_backend = f"ROCm / HIP ({torch.version.hip})"
        else:
            accel_backend = f"NVIDIA CUDA ({torch.version.cuda})"
    elif device.type == "mps":
        accel_backend = "Apple Metal Performance Shaders (MPS)"
        device_name = "Apple Silicon GPU"

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "Compute_Device": str(device),
        "acceleration_backend": accel_backend,
        "hardware_name": device_name,
        "cpu_threads": torch.get_num_threads(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "onnx_version": onnx.__version__,
        "onnxscript_version": onnxscript.__version__,
        "onnxruntime_version": ort.__version__,
        "ort_available_providers": ort.get_available_providers(),
    }


def run_tensor_sanity_check(device: torch.device) -> bool:
    """Ejecuta una multiplicación matricial rápida en el dispositivo detectado para validar operación."""
    torch.manual_seed(42)
    a = torch.randn((4, 4), device=device)
    b = torch.randn((4, 4), device=device)
    c = torch.matmul(a, b)
    return bool(c.shape == (4, 4) and not torch.isnan(c).any().item())


def print_diagnostic_report() -> torch.device:
    """Imprime el reporte completo de diagnóstico del nodo de entrenamiento y devuelve el dispositivo activo."""
    info = collect_environment_info()
    device = get_compute_device()
    sanity_ok = run_tensor_sanity_check(device)

    line = "=" * 72
    print(line)
    print("  NANO-AI | PC TRAINING NODE - REPORTE DE DIAGNÓSTICO (SEMANA 1)")
    print("  Rol R2 (Model Lead): David Isaí Portillo López")
    print(line)
    print(f"  [Sistema Operativo]     : {info['os']}")
    print(f"  [Versión de Python]     : {info['python_version']}")
    print("-" * 72)
    print(f"  [PyTorch Version]       : {info['torch_version']}")
    print(f"  [Backend Aceleración]   : {info['acceleration_backend']}")
    print(f"  [Dispositivo Seleccionado]: {info['Compute_Device']} ({info['hardware_name']})")
    print(f"  [Hilos de CPU PyTorch]  : {info['cpu_threads']}")
    print(f"  [Prueba Tensor MatMul]  : {'EXITOSA (OK)' if sanity_ok else 'FALLO'}")
    print("-" * 72)
    print(f"  [ONNX Version]          : {info['onnx_version']}")
    print(f"  [ONNX Script Version]   : {info['onnxscript_version']}")
    print(f"  [ONNX Runtime Version]  : {info['onnxruntime_version']}")
    print(f"  [ORT Execution Providers]: {', '.join(info['ort_available_providers'])}")
    print(f"  [NumPy / Scikit-Learn]  : {info['numpy_version']} / {info['sklearn_version']}")
    print(line)

    if not sanity_ok:
        raise RuntimeError("Fallo en la prueba de sanidad de tensores en PyTorch.")

    return device


if __name__ == "__main__":
    print_diagnostic_report()
