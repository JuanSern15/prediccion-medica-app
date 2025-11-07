from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

# Mapeo de diagnósticos a nombres descriptivos
DIAGNOSIS_NAMES = {
    1: "Dengue",
    2: "Malaria", 
    3: "Leptospirosis"
}

DIAGNOSIS_DESCRIPTIONS = {
    1: "Enfermedad viral transmitida por mosquitos Aedes",
    2: "Enfermedad parasitaria transmitida por mosquitos Anopheles",
    3: "Enfermedad bacteriana transmitida por agua o alimentos contaminados"
}

# Crear carpeta de uploads si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Cargar AMBOS modelos y componentes
print("Cargando modelos y componentes...")
model_lr = joblib.load('modelo_regresion_logistica.pkl')
model_nn = joblib.load('modelo_red_neuronal.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
model_info_lr = joblib.load('model_info_lr.pkl')
model_info_nn = joblib.load('model_info_nn.pkl')
print("✓ Modelos cargados exitosamente")
print(f"  - Regresión Logística: {model_info_lr['accuracy']*100:.2f}%")
print(f"  - Red Neuronal: {model_info_nn['accuracy']*100:.2f}%")

# Diccionario de modelos
MODELS = {
    'logistic': {
        'model': model_lr,
        'info': model_info_lr,
        'name': 'Regresión Logística'
    },
    'neural': {
        'model': model_nn,
        'info': model_info_nn,
        'name': 'Red Neuronal'
    }
}

# Valores promedio para campos opcionales (análisis de laboratorio)
# Estos valores se usan cuando el usuario no completa los campos opcionales
DEFAULT_VALUES = {
    # Hematología
    'hematocrit': 34.57,
    'hemoglobin': 11.46,
    'red_blood_cells': 4321049.38,
    'white_blood_cells': 5563.65,
    'neutrophils': 47.77,
    'eosinophils': 2.86,
    'basophils': 0.97,
    'monocytes': 12.90,
    'lymphocytes': 35.50,
    'platelets': 89276.75,
    # Química Sanguínea
    'AST (SGOT)': 140.58,
    'ALT (SGPT)': 97.36,
    'ALP (alkaline_phosphatase)': 137.48,
    'total_bilirubin': 1.58,
    'direct_bilirubin': 1.02,
    'indirect_bilirubin': 0.56,
    'total_proteins': 6.39,
    'albumin': 2.85,
    'creatinine': 0.87,
    'urea': 24.60
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def crear_matriz_confusion_imagen(y_true, y_pred):
    """Crea una imagen de la matriz de confusión normalizada"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalizar la matriz de confusión (proporción por fila, rango 0-1)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Etiquetas descriptivas
    labels = [DIAGNOSIS_NAMES[1], DIAGNOSIS_NAMES[2], DIAGNOSIS_NAMES[3]]
    
    plt.figure(figsize=(10, 8))
    
    # Mostrar la matriz normalizada con valores decimales
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=labels,
                yticklabels=labels,
                vmin=0, vmax=1,
                cbar_kws={'label': 'Proporción'})
    plt.title('Matriz de Confusión', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Diagnóstico Real', fontsize=12)
    plt.xlabel('Diagnóstico Predicho', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Convertir a base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', 
                         feature_names=feature_names,
                         model_lr_accuracy=f"{model_info_lr['accuracy']*100:.2f}",
                         model_nn_accuracy=f"{model_info_nn['accuracy']*100:.2f}")

@app.route('/predict_individual', methods=['POST'])
def predict_individual():
    """Predicción individual"""
    try:
        # Obtener el modelo seleccionado
        model_type = request.form.get('model_type', 'logistic')  # Por defecto: regresión logística
        
        if model_type not in MODELS:
            return jsonify({
                'success': False,
                'error': 'Modelo no válido'
            })
        
        selected_model = MODELS[model_type]['model']
        model_name = MODELS[model_type]['name']
        
        # Obtener datos del formulario
        data = {}
        for feature in feature_names:
            value = request.form.get(feature)
            # Si el campo está vacío Y está en DEFAULT_VALUES, usar valor por defecto
            if (value is None or value == '') and feature in DEFAULT_VALUES:
                data[feature] = DEFAULT_VALUES[feature]
            elif value is None or value == '':
                return jsonify({
                    'success': False,
                    'error': f'Falta el valor para: {feature}'
                })
            else:
                data[feature] = float(value)
        
        # Crear DataFrame
        df_input = pd.DataFrame([data])
        
        # Asegurar el orden correcto de las columnas
        df_input = df_input[feature_names]
        
        # Normalizar datos
        X_scaled = scaler.transform(df_input)
        
        # Hacer predicción con el modelo seleccionado
        prediction = selected_model.predict(X_scaled)[0]
        probabilities = selected_model.predict_proba(X_scaled)[0]
        
        # Preparar respuesta con nombres descriptivos
        result = {
            'success': True,
            'model_used': model_name,
            'diagnosis': int(prediction),
            'diagnosis_name': DIAGNOSIS_NAMES[int(prediction)],
            'diagnosis_description': DIAGNOSIS_DESCRIPTIONS[int(prediction)],
            'probabilities': {
                DIAGNOSIS_NAMES[1]: f'{probabilities[0]*100:.2f}%',
                DIAGNOSIS_NAMES[2]: f'{probabilities[1]*100:.2f}%',
                DIAGNOSIS_NAMES[3]: f'{probabilities[2]*100:.2f}%'
            },
            'confidence': f'{max(probabilities)*100:.2f}%'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predicción por lotes"""
    try:
        # Obtener el modelo seleccionado
        model_type = request.form.get('model_type', 'logistic')
        
        if model_type not in MODELS:
            return jsonify({
                'success': False,
                'error': 'Modelo no válido'
            })
        
        selected_model = MODELS[model_type]['model']
        model_name = MODELS[model_type]['name']
        
        # Verificar si se envió un archivo
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No se envió ningún archivo'
            })
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No se seleccionó ningún archivo'
            })
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Formato de archivo no permitido. Use .xlsx o .csv'
            })
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Leer archivo
        if filename.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
        
        # Verificar si tiene columna diagnosis (valores reales)
        has_true_labels = 'diagnosis' in df.columns
        
        if has_true_labels:
            y_true = df['diagnosis'].values
            X = df.drop('diagnosis', axis=1)
        else:
            y_true = None
            X = df
        
        # Agregar columnas faltantes con valores por defecto
        missing_cols = set(feature_names) - set(X.columns)
        if missing_cols:
            print(f"Columnas faltantes detectadas: {missing_cols}")
            print(f"Se usarán valores por defecto para: {len(missing_cols)} columnas")
            
            for col in missing_cols:
                if col in DEFAULT_VALUES:
                    # Usar valor por defecto del diccionario
                    X[col] = DEFAULT_VALUES[col]
                else:
                    # Si no está en DEFAULT_VALUES, usar 0
                    X[col] = 0
        
        # Asegurar orden correcto
        X = X[feature_names]
        
        # Normalizar y predecir con el modelo seleccionado
        X_scaled = scaler.transform(X)
        y_pred = selected_model.predict(X_scaled)
        y_proba = selected_model.predict_proba(X_scaled)
        
        # Preparar resultados
        results = {
            'success': True,
            'model_used': model_name,
            'total_predictions': len(y_pred),
            'predictions': []
        }
        
        # Agregar predicciones individuales
        for i in range(len(y_pred)):
            pred_detail = {
                'index': i + 1,
                'predicted_diagnosis': int(y_pred[i]),
                'predicted_name': DIAGNOSIS_NAMES[int(y_pred[i])],
                'probabilities': {
                    DIAGNOSIS_NAMES[1]: f'{y_proba[i][0]*100:.2f}%',
                    DIAGNOSIS_NAMES[2]: f'{y_proba[i][1]*100:.2f}%',
                    DIAGNOSIS_NAMES[3]: f'{y_proba[i][2]*100:.2f}%'
                }
            }
            if has_true_labels:
                pred_detail['true_diagnosis'] = int(y_true[i])
                pred_detail['true_name'] = DIAGNOSIS_NAMES[int(y_true[i])]
                pred_detail['correct'] = int(y_pred[i]) == int(y_true[i])
            
            results['predictions'].append(pred_detail)
        
        # Si hay etiquetas reales, calcular métricas
        if has_true_labels:
            accuracy = accuracy_score(y_true, y_pred)
            
            # Métricas por clase
            precision = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            # Métricas promediadas
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Matthews Correlation Coefficient
            mcc = matthews_corrcoef(y_true, y_pred)
            
            # Support (cantidad de muestras por clase en y_true)
            unique_classes, class_counts = np.unique(y_true, return_counts=True)
            support = {DIAGNOSIS_NAMES[cls]: int(count) for cls, count in zip(unique_classes, class_counts)}
            
            # Calcular especificidad por clase
            cm = confusion_matrix(y_true, y_pred)
            specificity = []
            for i in range(len(cm)):
                # True Negatives: suma de todas las celdas excepto la fila i y columna i
                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                # False Positives: suma de la columna i excepto el elemento diagonal
                fp = np.sum(cm[:, i]) - cm[i, i]
                specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            
            results['metrics'] = {
                'accuracy': f'{accuracy*100:.2f}%',
                'correct_predictions': int((y_true == y_pred).sum()),
                'incorrect_predictions': int((y_true != y_pred).sum()),
                'total_samples': len(y_true),
                'mcc': f'{mcc:.4f}',
                'macro_avg': {
                    'precision': f'{precision_macro*100:.2f}%',
                    'recall': f'{recall_macro*100:.2f}%',
                    'f1_score': f'{f1_macro*100:.2f}%'
                },
                'weighted_avg': {
                    'precision': f'{precision_weighted*100:.2f}%',
                    'recall': f'{recall_weighted*100:.2f}%',
                    'f1_score': f'{f1_weighted*100:.2f}%'
                },
                'by_class': {
                    DIAGNOSIS_NAMES[1]: {
                        'precision': f'{precision[0]*100:.2f}%',
                        'recall': f'{recall[0]*100:.2f}%',
                        'f1_score': f'{f1[0]*100:.2f}%',
                        'specificity': f'{specificity[0]*100:.2f}%',
                        'support': support.get(1, 0)
                    },
                    DIAGNOSIS_NAMES[2]: {
                        'precision': f'{precision[1]*100:.2f}%',
                        'recall': f'{recall[1]*100:.2f}%',
                        'f1_score': f'{f1[1]*100:.2f}%',
                        'specificity': f'{specificity[1]*100:.2f}%',
                        'support': support.get(2, 0)
                    },
                    DIAGNOSIS_NAMES[3]: {
                        'precision': f'{precision[2]*100:.2f}%',
                        'recall': f'{recall[2]*100:.2f}%',
                        'f1_score': f'{f1[2]*100:.2f}%',
                        'specificity': f'{specificity[2]*100:.2f}%',
                        'support': support.get(3, 0)
                    }
                }
            }
            
            # Crear matriz de confusión
            cm_image = crear_matriz_confusion_imagen(y_true, y_pred)
            results['confusion_matrix_image'] = cm_image
            
            # Matriz de confusión en formato tabla
            cm = confusion_matrix(y_true, y_pred)
            results['confusion_matrix'] = cm.tolist()
        
        # Distribución de predicciones
        unique, counts = np.unique(y_pred, return_counts=True)
        results['prediction_distribution'] = {
            DIAGNOSIS_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)
        }
        
        # Limpiar archivo temporal
        os.remove(filepath)
        
        return jsonify(results)
    
    except Exception as e:
        # Limpiar archivo si existe
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_feature_info')
def get_feature_info():
    """Devuelve información sobre las características del modelo"""
    return jsonify({
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'models': {
            'logistic': {
                'name': 'Regresión Logística',
                'accuracy': f"{model_info_lr['accuracy']*100:.2f}%"
            },
            'neural': {
                'name': 'Red Neuronal',
                'accuracy': f"{model_info_nn['accuracy']*100:.2f}%"
            }
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SISTEMA DE PREDICCIÓN DE DIAGNÓSTICO MÉDICO")
    print("="*60)
    print(f"Modelos cargados con {len(feature_names)} características")
    print(f"  - Regresión Logística: {model_info_lr['accuracy']*100:.2f}%")
    print(f"  - Red Neuronal: {model_info_nn['accuracy']*100:.2f}%")
    print("\nIniciando servidor Flask...")
    print("Accede a la aplicación en: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Usar puerto dinámico para producción (Render, Heroku, etc.)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
