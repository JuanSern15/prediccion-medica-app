// Tab switching
function openTab(evt, tabName) {
    const tabContents = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove("active");
    }
    
    const tabButtons = document.getElementsByClassName("tab-btn");
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove("active");
    }
    
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

// Reset form
function resetForm() {
    document.getElementById('individualForm').reset();
    document.getElementById('individualResult').style.display = 'none';
}

// Individual prediction
document.getElementById('individualForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const form = this;
    
    // Validar que al menos un síntoma principal esté marcado
    const symptomCheckboxes = form.querySelectorAll('input[type="checkbox"]');
    const hasSymptoms = Array.from(symptomCheckboxes).some(checkbox => checkbox.checked);
    
    if (!hasSymptoms) {
        alert('⚠️ Debe marcar al menos un síntoma principal para realizar la predicción.');
        return;
    }
    
    const formData = new FormData();
    
    // Procesar inputs normales (números)
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        formData.append(input.name, input.value);
    });
    
    // Procesar checkboxes (síntomas)
    const checkboxes = form.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        formData.append(checkbox.name, checkbox.checked ? '1' : '0');
    });
    
    // Procesar selects (género, origen, ocupación)
    const selects = form.querySelectorAll('select');
    selects.forEach(select => {
        const value = select.value;
        if (select.name === 'gender') {
            // Convertir a male/female binario
            formData.append('male', value === 'male' ? '1' : '0');
            formData.append('female', value === 'female' ? '1' : '0');
        } else if (select.name === 'origin') {
            // Convertir a urban/rural binario
            formData.append('urban_origin', value === 'urban' ? '1' : '0');
            formData.append('rural_origin', value === 'rural' ? '1' : '0');
        } else if (select.name === 'occupation') {
            // Convertir ocupación a one-hot encoding
            const occupations = ['homemaker', 'student', 'professional', 'merchant', 
                               'agriculture_livestock', 'various_jobs', 'unemployed'];
            occupations.forEach(occ => {
                formData.append(occ, value === occ ? '1' : '0');
            });
        }
    });
    
    // Procesar radio buttons (modelo)
    const modelRadio = form.querySelector('input[name="model_type"]:checked');
    if (modelRadio) {
        formData.append('model_type', modelRadio.value);
    }
    
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    // Loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="loading"></span> Analizando...';
    
    try {
        const response = await fetch('/predict_individual', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayIndividualResult(result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
});

function displayIndividualResult(result) {
    const resultContainer = document.getElementById('individualResult');
    const resultContent = document.getElementById('resultContent');
    
    let diagnosisClass = 'diagnosis-' + result.diagnosis;
    
    let html = `
        <div class="result-box">
            <h4 style="text-align: center; font-size: 1.4rem;">Resultado del Diagnóstico</h4>
            <div style="text-align: center; margin-bottom: 16px;">
                <span style="background: var(--gray-100); padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; color: var(--gray-700); display: inline-block;">
                    <i class="fas fa-brain"></i> Modelo: ${result.model_used}
                </span>
            </div>
            <div style="text-align: center; margin: 20px 0;">
                <span class="diagnosis-badge ${diagnosisClass}">
                    ${result.diagnosis_name}
                </span>
                <p style="color: #64748b; margin-top: 12px; font-size: 0.95rem;">
                    ${result.diagnosis_description}
                </p>
            </div>
        </div>
        
        <div class="result-box">
            <h4><i class="fas fa-chart-pie"></i> Probabilidades por Diagnóstico</h4>
            <div class="probability-bars" style="margin-top: 16px;">
    `;
    
    for (let [diagnosis, probability] of Object.entries(result.probabilities)) {
        let percent = parseFloat(probability);
        html += `
            <div class="probability-item">
                <div class="probability-label">
                    <span>${diagnosis}</span>
                    <span>${probability}</span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${percent}%"></div>
                </div>
            </div>
        `;
    }
    
    html += `
            </div>
        </div>
    `;
    
    resultContent.innerHTML = html;
    resultContainer.style.display = 'block';
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Batch file upload
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('batchFile');
const fileSelected = document.getElementById('fileSelected');
const fileName = document.getElementById('fileName');
const predictBtn = document.getElementById('predictBatchBtn');

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary)';
    uploadArea.style.background = 'rgba(79, 70, 229, 0.05)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--gray-300)';
    uploadArea.style.background = 'var(--gray-50)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--gray-300)';
    uploadArea.style.background = 'var(--gray-50)';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    const validExtensions = ['xlsx', 'csv'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    
    if (!validExtensions.includes(fileExtension)) {
        alert('Formato no válido. Use .xlsx o .csv');
        return;
    }
    
    // Guardar referencia al archivo
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
    
    fileName.textContent = file.name;
    uploadArea.style.display = 'none';
    fileSelected.style.display = 'flex';
    predictBtn.disabled = false;
}

function removeFile() {
    fileInput.value = '';
    uploadArea.style.display = 'block';
    fileSelected.style.display = 'none';
    predictBtn.disabled = true;
    document.getElementById('batchResult').style.display = 'none';
}

async function predictBatch() {
    // Verificar que hay un archivo seleccionado
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Por favor seleccione un archivo primero');
        return;
    }
    
    // Obtener el modelo seleccionado
    const modelType = document.querySelector('input[name="batch_model_type"]:checked').value;
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model_type', modelType);
    
    const btn = document.getElementById('predictBatchBtn');
    const originalText = btn.innerHTML;
    
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Procesando...';
    
    try {
        const response = await fetch('/predict_batch', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayBatchResult(result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

function displayBatchResult(result) {
    const resultContainer = document.getElementById('batchResult');
    const resultContent = document.getElementById('batchResultContent');
    
    let html = `
        <div class="result-box">
            <h4><i class="fas fa-info-circle"></i> Resumen</h4>
            ${result.model_used ? `
            <div class="result-item" style="margin-bottom: 12px;">
                <span class="badge" style="background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); padding: 8px 16px; border-radius: 8px; display: inline-flex; align-items: center; gap: 8px;">
                    <i class="fas fa-${result.model_used.includes('Neuronal') ? 'brain' : 'chart-line'}"></i>
                    <span style="font-weight: 600;">Modelo: ${result.model_used}</span>
                </span>
            </div>
            ` : ''}
            <div class="result-item">
                <span class="result-label">Total de Predicciones:</span>
                <span class="result-value">${result.total_predictions}</span>
            </div>
            <div style="margin-top: 16px;">
                <strong style="color: var(--gray-700);">Distribución:</strong>
                <div style="margin-top: 12px;">
    `;
    
    for (let [diagnosis, count] of Object.entries(result.prediction_distribution)) {
        let percent = (count / result.total_predictions * 100).toFixed(1);
        html += `
            <div class="probability-item">
                <div class="probability-label">
                    <span>${diagnosis}</span>
                    <span>${count} (${percent}%)</span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${percent}%"></div>
                </div>
            </div>
        `;
    }
    
    html += `
                </div>
            </div>
        </div>
    `;
    
    // Metrics
    if (result.metrics) {
        html += `
            <div class="result-box">
                <h4><i class="fas fa-chart-line"></i> Métricas Generales</h4>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Precisión (Accuracy)</h4>
                        <div class="metric-value">${result.metrics.accuracy}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Total de Muestras</h4>
                        <div class="metric-value">${result.metrics.total_samples}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Correctas</h4>
                        <div class="metric-value" style="color: var(--success);">
                            ${result.metrics.correct_predictions}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h4>MCC</h4>
                        <div class="metric-value" style="font-size: 1.5rem;">${result.metrics.mcc}</div>
                        <small style="color: #666;">Matthews Correlation</small>
                    </div>
                </div>
            </div>

            <div class="result-box">
                <h4><i class="fas fa-calculator"></i> Promedios de Métricas</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 2rem;">
                    <div>
                        <h5 style="color: var(--primary); margin-bottom: 1rem;">Macro Average (promedio simple)</h5>
                        <div class="metrics-grid" style="grid-template-columns: repeat(3, 1fr);">
                            <div class="metric-card">
                                <h4>Precision</h4>
                                <div class="metric-value" style="font-size: 1.3rem;">${result.metrics.macro_avg.precision}</div>
                            </div>
                            <div class="metric-card">
                                <h4>Recall</h4>
                                <div class="metric-value" style="font-size: 1.3rem;">${result.metrics.macro_avg.recall}</div>
                            </div>
                            <div class="metric-card">
                                <h4>F1-Score</h4>
                                <div class="metric-value" style="font-size: 1.3rem;">${result.metrics.macro_avg.f1_score}</div>
                            </div>
                        </div>
                    </div>
                    <div>
                        <h5 style="color: var(--primary); margin-bottom: 1rem;">Weighted Average (ponderado por soporte)</h5>
                        <div class="metrics-grid" style="grid-template-columns: repeat(3, 1fr);">
                            <div class="metric-card">
                                <h4>Precision</h4>
                                <div class="metric-value" style="font-size: 1.3rem;">${result.metrics.weighted_avg.precision}</div>
                            </div>
                            <div class="metric-card">
                                <h4>Recall</h4>
                                <div class="metric-value" style="font-size: 1.3rem;">${result.metrics.weighted_avg.recall}</div>
                            </div>
                            <div class="metric-card">
                                <h4>F1-Score</h4>
                                <div class="metric-value" style="font-size: 1.3rem;">${result.metrics.weighted_avg.f1_score}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="result-box">
                <h4><i class="fas fa-list-alt"></i> Métricas por Clase</h4>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem;">
        `;
        
        // Iterate through each class
        for (const [className, metrics] of Object.entries(result.metrics.by_class)) {
            html += `
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--primary);">
                    <h5 style="color: var(--primary); margin-bottom: 1rem; text-align: center;">${className}</h5>
                    <div style="display: grid; gap: 0.5rem;">
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: white; border-radius: 4px;">
                            <strong>Precision:</strong>
                            <span>${metrics.precision}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: white; border-radius: 4px;">
                            <strong>Recall:</strong>
                            <span>${metrics.recall}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: white; border-radius: 4px;">
                            <strong>F1-Score:</strong>
                            <span>${metrics.f1_score}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: white; border-radius: 4px;">
                            <strong>Specificity:</strong>
                            <span>${metrics.specificity}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: white; border-radius: 4px;">
                            <strong>Support:</strong>
                            <span style="color: var(--primary); font-weight: bold;">${metrics.support} muestras</span>
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += `
                </div>
            </div>
        `;
        
        if (result.confusion_matrix_image) {
            html += `
                <div class="result-box">
                    <h4><i class="fas fa-table"></i> Matriz de Confusión</h4>
                    <div class="confusion-matrix-container">
                        <img src="data:image/png;base64,${result.confusion_matrix_image}" alt="Matriz de Confusión">
                    </div>
                </div>
            `;
        }
    }
    
    resultContent.innerHTML = html;
    resultContainer.style.display = 'block';
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
