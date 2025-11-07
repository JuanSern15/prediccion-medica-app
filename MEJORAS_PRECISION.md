# Estrategias para Mejorar la Precisión del Modelo

## Situación Actual
- **Precisión**: 70.59% con 55 variables
- **Problema**: Dataset pequeño (81 pacientes) con desbalance de clases (51/19/11)
- **Modelos**: Regresión Logística y Red Neuronal

## Técnicas Implementadas

### 1. **Selección de Características (Feature Selection)**
- **Objetivo**: Eliminar variables irrelevantes que añaden ruido
- **Método**: ANOVA F-test para identificar las 30 variables más discriminativas
- **Beneficio**: Reduce overfitting y mejora generalización

### 2. **Balanceo de Clases con SMOTE**
- **Problema**: Dengue (51) >> Malaria (19) >> Leptospirosis (11)
- **Solución**: SMOTE-Tomek genera muestras sintéticas de clases minoritarias
- **Beneficio**: El modelo aprende mejor a identificar Malaria y Leptospirosis

### 3. **Optimización de Hiperparámetros (GridSearchCV)**

#### Regresión Logística:
```python
Parámetros a probar:
- C: [0.001, 0.01, 0.1, 1, 10, 100] → Regularización
- solver: ['lbfgs', 'saga', 'liblinear'] → Algoritmo de optimización
- max_iter: [500, 1000, 2000] → Iteraciones
- class_weight: ['balanced', None] → Penalización por desbalance
Total: 72 combinaciones
```

#### Red Neuronal:
```python
Parámetros a probar:
- hidden_layer_sizes: [(100,), (64,32), (128,64,32), (100,50,25)]
- activation: ['relu', 'tanh']
- alpha: [0.0001, 0.001, 0.01] → Regularización L2
- learning_rate: ['constant', 'adaptive']
- max_iter: [500, 1000]
Total: 80 combinaciones
```

### 4. **Validación Cruzada Estratificada**
- **Método**: 5-fold stratified cross-validation
- **Beneficio**: Evaluación más confiable que un solo train/test split
- **Resultado**: Media ± desviación estándar de precisión

### 5. **Modelo de Ensamble (Gradient Boosting)**
- **Método**: Combina múltiples árboles de decisión débiles
- **Parámetros**: 200 árboles
- **Beneficio**: Usualmente mejor que modelos lineales en datasets complejos

## Mejoras Esperadas

### Antes de Optimización:
```
Regresión Logística: 70.59%
Red Neuronal:        70.59%
```

### Después de Optimización (estimado):
```
Regresión Logística: 75-82% (con hiperparámetros óptimos + SMOTE)
Red Neuronal:        78-85% (con arquitectura óptima + SMOTE)
Gradient Boosting:   80-88% (modelo de ensamble)
```

## Otras Técnicas Avanzadas (Si aún no es suficiente)

### 6. **Ensemble Stacking**
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Combinar predicciones de múltiples modelos
ensemble = VotingClassifier(
    estimators=[
        ('lr', mejor_lr),
        ('nn', mejor_nn),
        ('gb', mejor_gb)
    ],
    voting='soft'  # Usa probabilidades
)
```

### 7. **Feature Engineering Avanzado**
```python
# Crear nuevas features combinando existentes
X['ratio_neutrophils_lymphocytes'] = X['neutrophils'] / (X['lymphocytes'] + 1)
X['platelets_severity'] = (X['platelets'] < 50000).astype(int)
X['liver_damage'] = ((X['AST (SGOT)'] > 100) | (X['ALT (SGPT)'] > 100)).astype(int)
```

### 8. **PCA (Principal Component Analysis)**
```python
from sklearn.decomposition import PCA

# Reducir dimensionalidad preservando varianza
pca = PCA(n_components=0.95)  # 95% de varianza
X_reduced = pca.fit_transform(X_scaled)
```

### 9. **XGBoost (Extreme Gradient Boosting)**
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    scale_pos_weight=3  # Para desbalance
)
```

### 10. **Aumentación de Datos (Data Augmentation)**
```python
from imblearn.over_sampling import ADASYN

# Más sofisticado que SMOTE
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)
```

## Limitaciones del Dataset

### Factores que limitan la precisión máxima:
1. **Tamaño**: Solo 81 pacientes (muy pequeño para ML)
2. **Desbalance**: 4.6x entre clase mayoritaria y minoritaria
3. **Solapamiento**: Las 3 enfermedades comparten síntomas similares
4. **Variabilidad biológica**: Pacientes con misma enfermedad tienen presentaciones diferentes

### Precisión realista máxima esperada:
- **Con 81 pacientes**: 80-90%
- **Con 500+ pacientes**: 90-95%
- **Con 2000+ pacientes**: 95-98%

## Recomendaciones

### Para este proyecto:
1. ✅ Usar optimización de hiperparámetros (GridSearchCV)
2. ✅ Aplicar SMOTE para balanceo
3. ✅ Seleccionar mejores features
4. ✅ Probar Gradient Boosting
5. ⭐ Si aún necesitas más: usar ensemble stacking

### Para trabajo futuro:
1. 📊 Recolectar más datos (objetivo: 200-500 pacientes)
2. 🔬 Agregar más análisis de laboratorio específicos
3. 🧬 Considerar datos genéticos o de imágenes médicas
4. 🤖 Explorar deep learning si el dataset crece a 1000+ muestras
