import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ENTRENAMIENTO CON BALANCEO DE DATOS (SMOTE)")
print("Regresión Logística y Red Neuronal")
print("=" * 80)

# 1. Cargar datos
print("\n1. Cargando datos...")
df = pd.read_excel("DEMALE-HSJM_2025_data.xlsx")
print(f"   ✓ Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")

# 2. Preparar todas las características (55 variables)
print(f"\n2. Preparando TODAS las características...")

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Codificar variables categóricas si existen
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    X[col] = pd.Categorical(X[col]).codes

print(f"   ✓ Total de características: {X.shape[1]}")
print(f"\n   Distribución ORIGINAL de clases (DESBALANCEADA):")
print(f"     - Dengue (1): {(y==1).sum()} pacientes ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"     - Malaria (2): {(y==2).sum()} pacientes ({(y==2).sum()/len(y)*100:.1f}%)")
print(f"     - Leptospirosis (3): {(y==3).sum()} pacientes ({(y==3).sum()/len(y)*100:.1f}%)")
print(f"     - Ratio de desbalance: {(y==1).sum() / (y==3).sum():.2f}x")

# Guardar nombres de características
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# 3. Dividir datos ANTES de SMOTE (importante!)
print("\n3. Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Entrenamiento: {X_train.shape[0]} muestras")
print(f"   ✓ Prueba: {X_test.shape[0]} muestras")

# 4. Normalizar
print("\n4. Normalizando datos...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')
print(f"   ✓ Scaler guardado")

# 5. APLICAR SMOTE SOLO AL CONJUNTO DE ENTRENAMIENTO
print("\n5. Aplicando SMOTE para balancear clases de ENTRENAMIENTO...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"   ✓ Dataset de entrenamiento BALANCEADO:")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for label, count in zip(unique, counts):
    diagnosis_name = {1: 'Dengue', 2: 'Malaria', 3: 'Leptospirosis'}[label]
    print(f"     - {diagnosis_name} ({label}): {count} muestras ({count/len(y_train_balanced)*100:.1f}%)")
print(f"   ✓ Total muestras de entrenamiento: {len(y_train_balanced)} (antes: {len(y_train)})")
print(f"   ⚠ Nota: El conjunto de PRUEBA permanece DESBALANCEADO (datos reales)")

# ============================================================================
# MODELO 1: REGRESIÓN LOGÍSTICA CON DATOS BALANCEADOS
# ============================================================================
print("\n" + "="*80)
print("MODELO 1: REGRESIÓN LOGÍSTICA (Entrenado con SMOTE)")
print("="*80)

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs',
    multi_class='multinomial'
)

lr_model.fit(X_train_balanced, y_train_balanced)
y_pred_lr = lr_model.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"✓ Precisión en test: {accuracy_lr*100:.2f}%")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_lr, 
                          target_names=['Dengue', 'Malaria', 'Leptospirosis'],
                          zero_division=0))

# Guardar modelo
joblib.dump(lr_model, 'modelo_regresion_logistica.pkl')
model_info_lr = {
    'accuracy': accuracy_lr,
    'n_features': len(feature_names),
    'feature_names': feature_names,
    'model_type': 'Regresión Logística',
    'solver': lr_model.solver,
    'balanced_with_smote': True
}
joblib.dump(model_info_lr, 'model_info_lr.pkl')
print("✓ Modelo Regresión Logística guardado")

# ============================================================================
# MODELO 2: RED NEURONAL (MLP) CON DATOS BALANCEADOS
# ============================================================================
print("\n" + "="*80)
print("MODELO 2: RED NEURONAL (MLP) (Entrenado con SMOTE)")
print("="*80)

nn_model = MLPClassifier(
    hidden_layer_sizes=(100, 50, 25),  # 3 capas ocultas
    activation='tanh',
    solver='adam',
    alpha=0.0001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

nn_model.fit(X_train_balanced, y_train_balanced)
y_pred_nn = nn_model.predict(X_test_scaled)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

print(f"✓ Precisión en test: {accuracy_nn*100:.2f}%")
print(f"✓ Iteraciones: {nn_model.n_iter_}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_nn, 
                          target_names=['Dengue', 'Malaria', 'Leptospirosis'],
                          zero_division=0))

# Guardar modelo
joblib.dump(nn_model, 'modelo_red_neuronal.pkl')
model_info_nn = {
    'accuracy': accuracy_nn,
    'n_features': len(feature_names),
    'feature_names': feature_names,
    'model_type': 'Red Neuronal',
    'hidden_layers': nn_model.hidden_layer_sizes,
    'n_iterations': nn_model.n_iter_,
    'balanced_with_smote': True
}
joblib.dump(model_info_nn, 'model_info_nn.pkl')
print("✓ Modelo Red Neuronal guardado")

# ============================================================================
# COMPARACIÓN FINAL
# ============================================================================
print("\n" + "="*80)
print("COMPARACIÓN DE MODELOS (Entrenados con SMOTE)")
print("="*80)
print(f"Dataset: {len(feature_names)} características (TODAS las variables)")
print(f"Balanceo: SMOTE aplicado al entrenamiento")
print(f"\nResultados en test (datos reales desbalanceados):")
print(f"  Regresión Logística:   {accuracy_lr*100:6.2f}%")
print(f"  Red Neuronal:          {accuracy_nn*100:6.2f}%")

best_model = max([
    ('Regresión Logística', accuracy_lr),
    ('Red Neuronal', accuracy_nn)
], key=lambda x: x[1])

print(f"\n🏆 Mejor modelo: {best_model[0]} ({best_model[1]*100:.2f}%)")

print("\n" + "="*70)
print("ENTRENAMIENTO COMPLETADO")
print("="*70)
print("\nArchivos generados:")
print("  • modelo_regresion_logistica.pkl")
print("  • modelo_red_neuronal.pkl")
print("  • model_info_lr.pkl")
print("  • model_info_nn.pkl")
print("  • scaler.pkl")
print("  • feature_names.pkl")
