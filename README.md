# Nano-AI — PC Training Node (Semana 1)

> **Módulo:** PC Training Node — Diseño, Entrenamiento Base y Exportación ONNX  
> **Proyecto:** Nano-AI (Sistema Híbrido PC Training Node $\rightarrow$ Raspberry Pi Edge Node)  
> **Rol:** R2 - Model Lead  
> **Responsable:** David Isaí Portillo López  

---

## 📋 Resumen del Módulo

Este repositorio contiene la implementación completa del nodo de entrenamiento (**PC Training Node**) para el proyecto **Nano-AI**, orientado al desarrollo y exportación de modelos ultracompactos entrenados **estrictamente desde cero (*Random Starting Weights*)** para su posterior despliegue en un nodo de borde (**Raspberry Pi 4 / 4 GB RAM**).

El flujo abarca:
1. **Diagnóstico automático de entorno y aceleración:** Detección de CUDA, ROCm/HIP, MPS y fallback determinista a CPU (`src/env_check.py`).
2. **Generador físico-estadístico del sensor DS18B20:** Series temporales sintéticas con ciclo térmico circadiano, armónico diurno y ruido estocástico suave (`src/synthetic_data.py`).
3. **Topologías neuronales ultra-ligeras:**
   - **`ForecasterRegressor` (Track 4):** MLP de 3 capas (`20 -> 32 -> 16 -> 1`) para predicción térmica a $+30\text{ min}$ (1,217 parámetros, $\sim 4.75\text{ KB}$).
   - **`AnomalyAutoencoder` (Track 3):** Autoencoder en espejo (`12 -> 8 -> 3 -> 8 -> 12`) con cuello de botella para detección no supervisada de anomalías por error de reconstrucción (271 parámetros, $\sim 1.06\text{ KB}$).
4. **Pipeline de entrenamiento y exportación ONNX:**
   - Estandarización persistida en JSON plano (`artifacts/scaler_synthetic.json`) para desacoplar el preprocesamiento en el nodo Edge sin dependencias de `pickle`.
   - Exportación a formato abierto `ONNX` (`opset_version=17`, con dimensión dinámica de batch).
   - Verificación de paridad numérica entre PyTorch y `onnxruntime.InferenceSession` ($\text{error máx} < 10^{-6}$).

---

## 📂 Estructura del Repositorio

```text
regression_training/
├── artifacts/
│   ├── autoencoder_skeleton.onnx      # Modelo ONNX del AnomalyAutoencoder
│   ├── forecaster_skeleton.onnx       # Modelo ONNX del ForecasterRegressor
│   ├── loss_curves_skeleton.png       # Gráficas de convergencia de entrenamiento
│   └── scaler_synthetic.json          # Medias y escalas de normalización (JSON plano)
├── docs/
│   └── R2_David_Architecture_and_Logbook_W1.md  # Arquitectura detallada y bitácora
├── src/
│   ├── __init__.py                    # Módulo Python del paquete src
│   ├── env_check.py                   # Diagnóstico de librerías y hardware
│   ├── models.py                      # Arquitecturas PyTorch con init Kaiming/Xavier
│   ├── synthetic_data.py              # Generador de datos sintéticos DS18B20
│   └── train_and_export_skeletons.py  # Pipeline integral de entrenamiento y exportación
├── .gitignore
├── requirements.txt                   # Dependencias principales del proyecto
└── README.md
```

---

## ⚙️ Especificaciones Técnicas de los Modelos

| Métrica / Parámetro | ForecasterRegressor (Track 4) | AnomalyAutoencoder (Track 3) |
| :--- | :---: | :---: |
| **Objetivo** | Predicción térmica continua a $+30\text{ min}$ | Detección no supervisada de anomalías |
| **Entrada ($D_{\text{in}}$)** | 20 ($18$ lecturas $+ \sin/\cos$ hora) | 12 ($10$ lecturas $+$ media móvil $+ \Delta T$) |
| **Salida ($D_{\text{out}}$)** | 1 (Escalar continuo de temperatura) | 12 (Reconstrucción del vector de entrada) |
| **Arquitectura** | `Linear(20, 32)` $\rightarrow$ `Linear(32, 16)` $\rightarrow$ `Linear(16, 1)` | `(12 -> 8 -> 3)` $\rightarrow$ `(3 -> 8 -> 12)` |
| **Inicialización** | Kaiming Normal (capas ReLU) + Xavier (salida) | Kaiming Normal (capas ReLU) + Xavier (salida) |
| **Parámetros Entrenables** | **1,217** | **271** |
| **Tamaño en Memoria (FP32)** | $\sim \mathbf{4.75\text{ KB}}$ | $\sim \mathbf{1.06\text{ KB}}$ |
| **Operaciones MAC por Inferencia** | $1,168$ MACs | $240$ MACs |
| **Formato de Exportación** | ONNX (Opset 17) | ONNX (Opset 17) |

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/Farafolodolo/regression_training.git
cd regression_training
```

### 2. Crear y activar entorno virtual
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar diagnóstico de entorno
```bash
python -m src.env_check
```

### 5. Ejecutar pipeline completo de entrenamiento y exportación
```bash
python -m src.train_and_export_skeletons
```

---

## 🎯 Ventajas para el Despliegue en Raspberry Pi
1. **Consumo despreciable de memoria RAM:** Los dos modelos juntos pesan $< 6\text{ KB}$ en tensores crudos y $< 25\text{ KB}$ en archivo ONNX, evitando cargar PyTorch en la Raspberry Pi y ahorrando $> 600\text{ MB}$ de RAM gracias a `onnxruntime`.
2. **Latencia Sub-milisegundo:** Ambas arquitecturas caben en la memoria caché L1 del procesador ARM Cortex-A72/A76, previniendo sobrecalentamiento en la CPU del dispositivo.
3. **Preprocesamiento Portable:** Al almacenar medias y desviaciones estándar en `scaler_synthetic.json`, el nodo de borde aplica $z = (x - \mu)/\sigma$ con operaciones nativas sin depender de `scikit-learn` ni `pickle`.
