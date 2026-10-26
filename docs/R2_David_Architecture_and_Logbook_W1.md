# Documento de Arquitectura de Modelos y Bitácora Semanal (Semana 1)

- **Proyecto:** Nano-AI (Sistema Híbrido PC Training Node $\rightarrow$ Raspberry Pi Edge Node)
- **Módulo:** PC Training Node — Diseño, Entrenamiento Base y Exportación ONNX
- **Responsable:** **David Isaí Portillo López**
- **Rol:** **R2 - Model Lead**
- **Fase:** Semana 1 — Configuración de Entorno, Datos Sintéticos DS18B20 y Esqueletos ONNX

---

## 1. Registro de Decisiones de Tamaño de Red y Topología Inicial

Para cumplir con las restricciones operativas del nodo embebido (Raspberry Pi con 4 GB de RAM compartida con el sistema operativo y servicios de telemetría) y garantizar inferencia determinista en tiempo real (< 1 ms por ventana), se diseñaron dos arquitecturas ultracompactas entrenadas **estrictamente desde cero** (*Random Starting Weights*).

### 1.1 Modelo 1: `ForecasterRegressor` (Track 4 — Predicción Térmica a +30 min)

El modelo recibe una ventana deslizante de **18 lecturas** del sensor DS18B20 muestreadas cada 10 segundos (equivalente a 3 minutos de historia inmediata) concatenada con la codificación cíclica trigonométrica de la hora del día (`sin_hora`, `cos_hora`), sumando **20 características de entrada**, y predice un único escalar continuo: la temperatura ambiente exacta dentro de 30 minutos ($+180$ pasos de 10 segundos).

| Capa | Tipo de Operación | Forma de Entrada | Forma de Salida | Función de Activación | Inicialización de Pesos | Pesos ($W$) | Sesgos ($b$) | Total Parámetros |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fc1` | `nn.Linear(20, 32)` | `(N, 20)` | `(N, 32)` | `ReLU` | Kaiming Normal (`fan_in`) | $20 \times 32 = 640$ | $32$ | **672** |
| `fc2` | `nn.Linear(32, 16)` | `(N, 32)` | `(N, 16)` | `ReLU` | Kaiming Normal (`fan_in`) | $32 \times 16 = 512$ | $16$ | **528** |
| `fc_out` | `nn.Linear(16, 1)` | `(N, 16)` | `(N, 1)` | Lineal (Identidad) | Xavier Uniform (Glorot) | $16 \times 1 = 16$ | $1$ | **17** |
| **Total** | **MLP 3 Capas** | **`(N, 20)`** | **`(N, 1)`** | — | — | **1,168** | **49** | **1,217** |

- **Parámetros entrenables exactos:** `1,217` parámetros.
- **Huella en memoria de parámetros (FP32):** $1,217 \times 4\text{ bytes} = 4,868\text{ bytes} \approx \mathbf{4.75\text{ KB}}$.

---

### 1.2 Modelo 2: `AnomalyAutoencoder` (Track 3 — Detección de Anomalías por Reconstrucción)

El modelo opera bajo un paradigma **no supervisado / semi-supervisado** entrenándose exclusivamente con lecturas de régimen térmico normal. Recibe un vector de **12 características** conformado por las últimas 10 lecturas del sensor DS18B20, la media móvil de dicha ventana (`moving_avg_10`) y la derivada discreta inmediata (`delta_t` $= T_t - T_{t-1}$). Su topología en espejo comprime la señal en un espacio latente de **3 neuronas** (*bottleneck*) y reconstruye las 12 variables originales.

| Bloque | Capa | Tipo de Operación | Forma de Entrada | Forma de Salida | Función de Activación | Inicialización de Pesos | Pesos ($W$) | Sesgos ($b$) | Total Parámetros |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Encoder** | `encoder.0` | `nn.Linear(12, 8)` | `(N, 12)` | `(N, 8)` | `ReLU` | Kaiming Normal (`fan_in`) | $12 \times 8 = 96$ | $8$ | **104** |
| **Bottleneck** | `encoder.2` | `nn.Linear(8, 3)` | `(N, 8)` | `(N, 3)` | `ReLU` | Kaiming Normal (`fan_in`) | $8 \times 3 = 24$ | $3$ | **27** |
| **Decoder** | `decoder_hidden` | `nn.Linear(3, 8)` | `(N, 3)` | `(N, 8)` | `ReLU` | Kaiming Normal (`fan_in`) | $3 \times 8 = 24$ | $8$ | **32** |
| **Salida** | `decoder_out` | `nn.Linear(8, 12)` | `(N, 8)` | `(N, 12)` | Lineal (Identidad) | Xavier Uniform (Glorot) | $8 \times 12 = 96$ | $12$ | **108** |
| **Total** | — | **Autoencoder Simétrico** | **`(N, 12)`** | **`(N, 12)`** | — | — | **240** | **31** | **271** |

- **Parámetros entrenables exactos:** `271` parámetros.
- **Huella en memoria de parámetros (FP32):** $271 \times 4\text{ bytes} = 1,084\text{ bytes} \approx \mathbf{1.06\text{ KB}}$.

---

## 2. Justificación Técnica para la Defensa del Demo

### 2.1 ¿Por qué esta arquitectura es ideal para los 4 GB de RAM de la Raspberry Pi?
1. **Huella de Memoria Despreciable (< 6 KB en tensores FP32):**
   Ambos modelos combinados suman apenas **1,488 parámetros** ($\sim 5.81\text{ KB}$ en memoria cruda y $< 25\text{ KB}$ empaquetados con el grafo ONNX completo). En una Raspberry Pi con 4 GB de RAM, donde el sistema operativo Linux, el colector de hardware 1-Wire del DS18B20 y la interfaz de monitoreo consumen parte importante de la memoria, ejecutar ambos modelos mediante `onnxruntime` (`CPUExecutionProvider`) evita por completo instalar PyTorch completo en el dispositivo embebido (ahorrando $> 600\text{ MB}$ de RAM en runtime).
2. **Complejidad Computacional $O(1)$ Sub-milisegundo y Sin Estrangulamiento Térmico:**
   El paso hacia adelante (*forward pass*) requiere únicamente $1,168$ operaciones MAC (*Multiply-Accumulate*) para el regresor y $240$ operaciones MAC para el autoencoder. Esto cabe íntegramente en la caché L1 del procesador ARM Cortex-A72/A76 de la Raspberry Pi, logrando latencias de inferencia en el orden de **microsegundos** sin elevar la temperatura de la propia CPU (lo cual es crítico para no contaminar térmicamente las lecturas cercanas del sensor DS18B20).
3. **Estandarización Desacoplada vía JSON (`scaler_synthetic.json`):**
   Al exportar las medias (`mean_`) y desviaciones estándar (`scale_`) en un archivo JSON plano en lugar de usar `pickle` o `joblib`, el nodo Edge aplica la transformación $z = \frac{x - \mu}{\sigma}$ con operaciones vectoriales nativas sin dependencias pesadas ni riesgos de incompatibilidad de versiones de `scikit-learn` entre arquitecturas `x86_64` (PC) y `aarch64` (Raspberry Pi).

### 2.2 ¿Por qué se entrena desde pesos aleatorios (*Random Starting Weights*)?
1. **Cumplimiento Normativo y Pureza Experimental:**
   El proyecto Nano-AI exige desarrollar modelos entrenados íntegramente desde cero sin transferencia de aprendizaje (*transfer learning*) ni pesos pre-entrenados externos.
2. **Inicialización Matemática Adaptada a la Topología (Kaiming He + Xavier Glorot):**
   - En las capas ocultas con activación `ReLU`, se aplica **Kaiming Normal** ($\mathcal{N}(0, \sqrt{2 / n_{\text{in}}})$), compensando la anulación de la mitad negativa del espectro que introduce `ReLU` y evitando que la varianza de las activaciones colapse o explote desde la época 1.
   - En las capas de salida lineal (`fc_out` y `decoder_out`), se aplica **Xavier Uniform** ($\mathcal{U}(-\sqrt{6/(n_{\text{in}}+n_{\text{out}})}, \sqrt{6/(n_{\text{in}}+n_{\text{out}})})$), preservando la simetría de gradientes en regresión continua sobre datos estandarizados ($\mu=0, \sigma=1$).
3. **Prevención de Sobreparametrización:**
   Dado que la física térmica ambiental sigue ecuaciones diferenciales suaves (inercia térmica + ciclo circadiano solar de 24 h), una red compacta inicializada con Kaiming/Xavier converge rápidamente en pocas épocas sin memorizar el ruido blanco del sensor DS18B20.

### 2.3 ¿Por qué el Autoencoder usa un cuello de botella de 3 neuronas en lugar de un clasificador supervisado?
1. **Ausencia de Anomalías Etiquetadas en el Mundo Real:**
   Un clasificador supervisado binario (`Normal` vs. `Anomalía`) requiere ejemplos etiquetados de **todos los modos de fallo posibles** durante el entrenamiento (picos por contacto físico con el sensor, desconexión intermitente del bus 1-Wire, fuente de calor externa súbita, deriva de voltaje). En un entorno real con el DS18B20, las anomalías son eventos raros e impredecibles; cualquier anomalía no vista en el set de entrenamiento sería clasificada erróneamente por un clasificador supervisado.
2. **Compresión en Variedad de Baja Dimensión ($\mathbb{R}^{12} \rightarrow \mathbb{R}^{3} \rightarrow \mathbb{R}^{12}$):**
   En régimen normal, las 12 variables de entrada están altamente correlacionadas y gobernadas por solo **3 grados de libertad físicos fundamentales**:
   - Nivel térmico base de la ventana (capturado por la media local),
   - Pendiente o tendencia térmica local ($\Delta T / \Delta t$), y
   - Micro-curvatura suave de transición.
   El cuello de botella de **3 neuronas** fuerza a la red a aprender únicamente esta variedad (*manifold*) física normal, descartando grados de libertad espurios.
3. **Detección Universal mediante Error de Reconstrucción (MSE):**
   Cuando ocurre una anomalía (por ejemplo, un salto abrupto en `delta_t` o una lectura inconsistente con la media móvil de las 10 muestras), el vector de entrada cae fuera de la variedad aprendida. El cuello de botella de 3 neuronas es incapaz de comprimir y reconstruir esa perturbación atípica, disparando inmediatamente el **Error Cuadrático Medio de Reconstrucción**:
   $$\text{MSE}_{\text{recon}}(\mathbf{x}) = \frac{1}{12} \sum_{j=1}^{12} \left(x_j - \hat{x}_j\right)^2 > \tau$$
   Esto proporciona tanto una bandera de alerta inmediata (al superar el umbral estadístico $\tau$) como una métrica continua de severidad de la anomalía.

---

## 3. Bitácora Semanal (*Contribution Log — Semana 1*)

| Campo | Detalle |
| :--- | :--- |
| **Nombre del Integrante** | **David Isaí Portillo López** |
| **Rol en el Equipo** | **R2 — Model Lead (PC Training Node)** |
| **Semana de Trabajo** | Semana 1 |
| **Estado de Entregables** | Completado y Validado (100%) |

### Actividades Realizadas en la Semana 1:
1. **Configuración y Diagnóstico del Nodo de Entrenamiento (`src/env_check.py`):**
   - Implementación del verificador automático de hardware (`cuda` / `rocm` / `mps` con *fallback* limpio a `cpu`) y validación de versiones del ecosistema `torch`, `onnx`, `onnxscript`, `onnxruntime`, `numpy` y `scikit-learn`.
2. **Simulador Físico-Estadístico del Sensor DS18B20 (`src/synthetic_data.py`):**
   - Generación de series temporales sintéticas con ciclo térmico diario realista ($20\,^\circ\text{C}$ a $28\,^\circ\text{C}$) más armónico diurno y ruido gaussiano calibrado a la resolución del sensor digital DS18B20.
   - Estructuración de ventanas deslizantes `X_reg` `(N, 20)` con codificación cíclica `hour_sin`/`hour_cos` y horizonte `y_reg` `(N, 1)` a $+30\text{ min}$, así como ventanas `X_ae` `(N, 12)` con media móvil y `delta_t`.
3. **Diseño de Arquitecturas Neuronales desde Cero (`src/models.py`):**
   - Construcción de `ForecasterRegressor` (`20 -> 32 -> 16 -> 1`, 1,217 parámetros, 4.75 KB) y `AnomalyAutoencoder` (`12 -> 8 -> 3 -> 8 -> 12`, 271 parámetros, 1.06 KB) con inicialización explícita Kaiming Normal y Xavier Uniform.
4. **Pipeline de Entrenamiento, Exportación ONNX y Paridad Numérica (`src/train_and_export_skeletons.py`):**
   - Normalización `StandardScaler` persistida en `artifacts/scaler_synthetic.json`.
   - Entrenamiento de validación (30 épocas con `Adam` y `MSELoss`) y generación de curvas de convergencia en `artifacts/loss_curves_skeleton.png`.
   - Compilación a formato estándar `ONNX` (`opset_version=17` y `dynamic_axes` para `batch_size`) generando `artifacts/forecaster_skeleton.onnx` y `artifacts/autoencoder_skeleton.onnx`.
   - Verificación estructural con `onnx.checker.check_model` y prueba de inferencia cruzada contra `onnxruntime.InferenceSession` confirmando error numérico máximo $< 10^{-6}$ frente a PyTorch.

**Firma de Conformidad:**
> **David Isaí Portillo López**
> *R2 - Model Lead | Proyecto Nano-AI — Semana 1*
