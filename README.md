# 📊 Telecom X – Parte 2: Predicción de Cancelación (Churn)
## ⚙️ Instalación rápida
1) (Opcional) crear un entorno virtual
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```
2) Instalar dependencias mínimas
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```


---


## ▶️ Ejecución
1. Inicia Jupyter y abre el notebook:
```bash
jupyter notebook
```
2. Abre `Challenge_TelecomX_Parte2_VF.ipynb` y **ejecuta las celdas en orden**. El flujo incluye:
- **Split Train/Test** (estratificado) y/o **K-Fold**.
- **Preprocesamiento** con `ColumnTransformer`:
- `OneHotEncoder` para columnas categóricas (p. ej., `Contract`, `PaymentMethod`, `InternetService`).
- `StandardScaler` o `MinMaxScaler` para numéricas (`tenure`, `chargesMonthly`, `chargesTotal`, `cuentas_diarias`, etc.).
- **Balanceo de clases** (si corresponde): `SMOTE` / `NearMiss` / `class_weight`.
- **Entrenamiento** de varios clasificadores (p. ej., *LogisticRegression*, *KNeighborsClassifier*, *DecisionTreeClassifier*, *RandomForestClassifier*).
- **Evaluación** (métricas, validación cruzada, matriz de confusión, curvas ROC/PR si aplica).
- **Interpretación** (importancias, *permutation importance* o *SHAP* si se decide extender).
- **Serialización** del mejor modelo (guardar en `modelos/`).


---


## 🔄 Pipeline de Modelado (resumen)
1. **Carga & split** → Train/Test estratificado.
2. **Preprocesamiento** → Imputación de nulos, codificación, escalado.
3. **Balanceo** (opcional) → *SMOTE* / *undersampling* / `class_weight`.
4. **Modelos** → ≥ 2 algoritmos comparados con misma partición y *pipeline* consistente.
5. **Validación** → *k-fold* estratificado; reportar media y desviación estándar.
6. **Selección** → Mejor relación *recall (churn)* / *precision* / *f1* según objetivos.
7. **Explicabilidad** → Importancias y factores clave.
8. **Serialización & checklist** → Guardar modelo y columnas/transformers.


---


## 📊 Métricas a Reportar
- **Accuracy** (desempeño global)
- **Recall de la clase churn** (sensibilidad) → *prioritaria* para no perder cancelaciones reales
- **Precision de la clase churn** → controlar falsos positivos en campañas
- **F1-score** → balance precisión/recobrado
- **Matriz de confusión** → errores tipo I/II
- (Opcional) **ROC-AUC** y **PR-AUC** con probabilidades (`predict_proba`)


---


## 🧠 Interpretación e Insights (guía)
- Revisar importancias del modelo ganador: suelen destacar **tipo de contrato**, **método de pago**, **InternetService**, **tenure**, **cargos mensuales/totales** y servicios como **OnlineSecurity/TechSupport**.
- Contrastar con hallazgos del EDA (Parte 1) para coherencia.
- Documentar variables que **aumentan** o **reducen** el riesgo de churn y por qué.


---


## 🚀 Recomendaciones Estratégicas (ejemplos)
- **Segmentación** por probabilidad de churn y valor del cliente (priorizar alto riesgo/alto valor).
- **Campañas de retención** para alto riesgo: descuentos, upgrades, bundles de servicios.
- **Mejora del onboarding** y beneficios por permanencia para clientes de baja antigüedad.
- **Monitoreo** continuo de variables críticas (p. ej., spikes en `chargesMonthly`).


---


## ❗ Problemas frecuentes y soluciones
- **Desbalance de clases** → aplicar `SMOTE`, `NearMiss` o `class_weight`.
- **Sobreajuste** → *cross-validation*, regularización, *early stopping* (si se usa boosting), más datos.
- **Data leakage** → hacer *split* antes del preprocesamiento y usar `Pipeline/ColumnTransformer`.
- **Variabilidad de métricas** → fijar `random_state`, aumentar K en CV, usar intervalos de confianza.
