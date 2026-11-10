# Guía Rápida de Uso e Inferencia en Raspberry Pi (Modelos ONNX)

**Archivos necesarios para correr en la Raspberry Pi:**
1. `forecaster_real.onnx` — Modelo predictor de temperatura a +30 minutos.
2. `autoencoder_real.onnx` — Modelo detector de perturbaciones/anomalías térmicas.
3. `scaler_real.json` — Parámetros fijos de normalización (`x_mean`, `x_scale`, `y_mean`, `y_scale`) y el umbral calibrado de anomalía (`anomaly_mse_threshold`).

> **Importante:** En la Raspberry Pi **no se debe entrenar ni ajustar (`fit`) ningún `StandardScaler` nuevo**. Ambos modelos `.onnx` están calibrados matemáticamente con las medias y desviaciones estándar guardadas en `scaler_real.json`.

---

## 1. ¿Cómo funcionan y cómo te comunicas con cada modelo?

Los archivos `.onnx` funcionan como cajas matemáticas vectoriales: reciben un arreglo de números decimales (`float32`) en un orden específico y devuelven otro arreglo de números.

### A. Modelo 1: `forecaster_real.onnx` (Predicción a +30 min)
* **Nombre del tensor de entrada:** `"sensor_window_and_time"` — Forma `(1, 20)` (`float32`).
* **¿Qué le envías (20 números en orden)?**
  1. Las **últimas 18 temperaturas** leídas del sensor DS18B20 en `°C`: `[T_1, T_2, ..., T_18]` (donde `T_18` es la lectura más reciente).
  2. La hora actual codificada con seno y coseno:
     * `hour_sin = sin(2 * pi * hora_decimal / 24.0)`
     * `hour_cos = cos(2 * pi * hora_decimal / 24.0)`
* **Normalización antes de entrar al modelo:**
  $$\text{entrada\_norm} = \frac{\text{vector\_20} - \text{x\_mean}}{\text{x\_scale}}$$
* **¿Qué te devuelve?**
  Un único número normalizado (`shape (1, 1)`). Para convertirlo a **grados Celsius (`°C`) reales dentro de 30 minutos**, aplicas la des-normalización con `y_scale` y `y_mean` del JSON:
  $$T_{+30\text{ min}}\,(^\circ\text{C}) = (\text{salida\_onnx} \times \text{y\_scale}) + \text{y\_mean}$$

---

### B. Modelo 2: `autoencoder_real.onnx` (Detección de Anomalías / Perturbaciones)
* **Nombre del tensor de entrada:** `"normal_window_features"` — Forma `(1, 12)` (`float32`).
* **¿Qué le envías (12 números en orden)?**
  1. Las **últimas 10 temperaturas** en `°C`: `[T_1, T_2, ..., T_10]` (donde `T_10` es la más reciente).
  2. El **promedio (`media_movil`)** de esas 10 temperaturas: `mean(T_1 ... T_10)`.
  3. El **cambio inmediato (`delta_t`)** entre las últimas dos lecturas: `T_10 - T_9`.
* **Normalización antes de entrar al modelo:**
  $$\text{entrada\_ae\_norm} = \frac{\text{vector\_12} - \text{x\_mean}}{\text{x\_scale}}$$
* **¿Qué te devuelve y cómo detectas la anomalía?**
  El Autoencoder intenta **copiar/reconstruir** esos 12 números pasando por un embudo interno de solo 3 neuronas que únicamente conoce temperaturas normales:
  * Te devuelve **12 números reconstruidos** (`shape (1, 12)`).
  * Calculas el **Error Cuadrático Medio (`MSE`)** entre los 12 números normalizados que metiste y los 12 números que te devolvió:
    $$\text{MSE} = \text{mean}\left((\text{entrada\_ae\_norm} - \text{reconstruccion\_onnx})^2\right)$$
  * Si **`MSE > anomaly_mse_threshold`** (`0.66528`), significa que el modelo no pudo reconstruir la lectura porque sale del comportamiento normal $\rightarrow$ **¡Hay perturbación / anomalía!**
  * Si **`MSE <= anomaly_mse_threshold`**, el comportamiento es **Normal**.

---

## 2. Ejemplo Mínimo de Uso en Python

Dependencias únicas en la Raspberry Pi:
```bash
pip install numpy onnxruntime
```

Ejemplo listo para integrar:

```python
import json
import numpy as np
import onnxruntime as ort

# 1. Al iniciar el programa: cargar el JSON del scaler y abrir las 2 sesiones ONNX
with open("scaler_real.json", "r", encoding="utf-8") as f:
    scaler = json.load(f)

sess_reg = ort.InferenceSession("forecaster_real.onnx", providers=["CPUExecutionProvider"])
sess_ae  = ort.InferenceSession("autoencoder_real.onnx", providers=["CPUExecutionProvider"])

# -------------------------------------------------------------------------
# 2. En cada ciclo de lectura (cuando ya tengas las últimas 18 temperaturas):
# -------------------------------------------------------------------------
ultimas_18_temps = [
    24.50, 24.50, 24.50, 24.56, 24.56, 24.56,
    24.62, 24.62, 24.62, 24.62, 24.68, 24.68,
    24.68, 24.75, 24.75, 24.75, 24.81, 24.81
]
hora_decimal = 18.5  # Ejemplo: 18:30 hrs (6:30 PM)

# === A) PREDICCIÓN DE TEMPERATURA A +30 MINUTOS ===
angulo = 2.0 * np.pi * (hora_decimal / 24.0)
vector_20 = np.array(ultimas_18_temps + [np.sin(angulo), np.cos(angulo)], dtype=np.float32)

# Normalizar entrada con los valores del JSON -> shape (1, 20)
x_reg_mean  = np.array(scaler["forecaster_regressor"]["x_mean"], dtype=np.float32)
x_reg_scale = np.array(scaler["forecaster_regressor"]["x_scale"], dtype=np.float32)
entrada_reg = ((vector_20 - x_reg_mean) / x_reg_scale).reshape(1, 20)

# Inferencia ONNX y conversión a °C reales
salida_norm = sess_reg.run(None, {"sensor_window_and_time": entrada_reg})[0][0, 0]
y_mean  = scaler["forecaster_regressor"]["y_mean"][0]
y_scale = scaler["forecaster_regressor"]["y_scale"][0]
temp_pred_30m = (salida_norm * y_scale) + y_mean

print(f"Temperatura estimada en +30 min: {temp_pred_30m:.2f} °C")

# === B) DETECCIÓN DE ANOMALÍA / PERTURBACIÓN ===
ultimas_10 = ultimas_18_temps[-10:]
media_10   = float(np.mean(ultimas_10))
delta_t    = float(ultimas_10[-1] - ultimas_10[-2])
vector_12  = np.array(ultimas_10 + [media_10, delta_t], dtype=np.float32)

# Normalizar entrada con los valores del JSON -> shape (1, 12)
x_ae_mean  = np.array(scaler["anomaly_autoencoder"]["x_mean"], dtype=np.float32)
x_ae_scale = np.array(scaler["anomaly_autoencoder"]["x_scale"], dtype=np.float32)
entrada_ae = ((vector_12 - x_ae_mean) / x_ae_scale).reshape(1, 12)

# Inferencia ONNX y cálculo del error de reconstrucción (MSE)
reconstruccion_ae = sess_ae.run(None, {"normal_window_features": entrada_ae})[0]
mse_error  = float(np.mean((entrada_ae - reconstruccion_ae) ** 2))
umbral_tau = scaler["anomaly_autoencoder"]["anomaly_mse_threshold"]

es_anomalia = mse_error > umbral_tau
print(f"MSE Autoencoder: {mse_error:.4f} | Umbral: {umbral_tau:.4f} | ¿Anomalía?: {es_anomalia}")
```
