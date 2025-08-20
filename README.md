# 📊 Telecom X – Parte 2: Predicción de Cancelación (Churn)
## 📌 Propósito del Proyecto
Este proyecto corresponde a la **segunda fase del desafío Telecom X**. Tras el análisis exploratorio realizado en la Parte 1, aquí se desarrolla un **pipeline de Machine Learning** cuyo objetivo principal es **predecir la cancelación de clientes (churn)** a partir de variables relevantes. 

El propósito central es anticipar qué clientes tienen mayor probabilidad de cancelar su servicio, para que la empresa pueda diseñar estrategias de retención efectivas.

---

## 📂 Estructura del Proyecto
```
📦 telecomx-churn-part2
├── Challenge_TelecomX_Parte2_VF.ipynb   # Notebook principal con el pipeline de ML
├── data/                                # Carpeta con datasets crudos y tratados en CSV
├── visualizaciones/                     # (opcional) gráficos generados durante el análisis
├── modelos/                             # (opcional) modelos serializados .pkl/.joblib
└── README.md                            # Este archivo
```

---

## 🔄 Preparación de Datos
1. **Clasificación de variables:**
   - **Numéricas:** `tenure`, `ChargesMonthly`, `ChargesTotal`, `cuentas_diarias`, `SeniorCitizen`.
   - **Categóricas:** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.

2. **Normalización y codificación:**
   - `OneHotEncoder` aplicado a las variables categóricas.
   - `StandardScaler` o `MinMaxScaler` aplicado a las variables numéricas.

3. **Separación de datos:**
   - División en conjuntos de entrenamiento y prueba mediante `train_test_split` (estratificado para preservar la proporción de churn).

4. **Decisiones de modelización:**
   - Se exploraron diferentes algoritmos de clasificación (Regresión Logística, Árboles de Decisión, Random Forest, Gradient Boosting).
   - En caso de desbalance en la variable objetivo, se aplicaron técnicas como **SMOTE** o parámetros de `class_weight`.
   - Las métricas principales consideradas fueron *recall* y *f1-score* para la clase churn, priorizando la capacidad de detectar clientes en riesgo.

---

## ▶️ Instrucciones de Ejecución
1. **Clonar o descargar el repositorio.**

2. **Instalar librerías necesarias:**
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

3. **Abrir el notebook principal:**
```bash
jupyter notebook Challenge_TelecomX_Parte2_VF.ipynb
```

4. **Ejecutar las celdas en orden**, siguiendo el flujo de:
   - Carga de datos tratados en CSV.
   - Preprocesamiento de variables.
   - División train/test.
   - Entrenamiento de modelos.
   - Evaluación de métricas y visualizaciones.
   - Interpretación de resultados y conclusión.

---

## 📈 Resultados y Métricas Alcanzadas
Durante la experimentación se obtuvieron los siguientes resultados aproximados:
- **Regresión Logística:** Accuracy ≈ 0.80, Recall ≈ 0.68, F1-Score ≈ 0.72
- **Random Forest:** Accuracy ≈ 0.85, Recall ≈ 0.72, F1-Score ≈ 0.76
- **Gradient Boosting:** Accuracy ≈ 0.86, Recall ≈ 0.74, F1-Score ≈ 0.78
---


## 🧠 Interpretación e Insights (guía)
- Revisar importancias del modelo ganador: suelen destacar **tipo de contrato**, **método de pago**, **InternetService**, **tenure**, **cargos mensuales/totales** y servicios como **OnlineSecurity/TechSupport**.
- Contrastar con hallazgos del EDA (Parte 1) para coherencia.
- Documentar variables que **aumentan** o **reducen** el riesgo de churn y por qué.


## 🚀 Recomendaciones Estratégicas (ejemplos)
- **Segmentación** por probabilidad de churn y valor del cliente (priorizar alto riesgo/alto valor).
- **Campañas de retención** para alto riesgo: descuentos, upgrades, bundles de servicios.
- **Mejora del onboarding** y beneficios por permanencia para clientes de baja antigüedad.
- **Monitoreo** continuo de variables críticas (p. ej., spikes en `chargesMonthly`).


## ❗ Problemas frecuentes y soluciones
- **Desbalance de clases** → aplicar `SMOTE`, `NearMiss` o `class_weight`.
- **Sobreajuste** → *cross-validation*, regularización, *early stopping* (si se usa boosting), más datos.
- **Data leakage** → hacer *split* antes del preprocesamiento y usar `Pipeline/ColumnTransformer`.
- **Variabilidad de métricas** → fijar `random_state`, aumentar K en CV, usar intervalos de confianza.
