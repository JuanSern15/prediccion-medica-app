# 🚀 Guía de Deployment - Sistema de Predicción Médica

## Opción 1: Render.com (RECOMENDADO - GRATIS)

### Paso 1: Preparar el Repositorio de GitHub

1. **Crear cuenta en GitHub** (si no tienes): https://github.com
2. **Crear un nuevo repositorio:**
   - Nombre: `prediccion-medica-app` (o el que prefieras)
   - Público o Privado (ambos funcionan)
   - NO inicializar con README (ya lo tenemos)

3. **Subir el proyecto a GitHub:**
   ```bash
   # Abrir PowerShell en la carpeta del proyecto
   cd "c:\Users\Lenovo\Desktop\Proyecto final Analitica"
   
   # Inicializar Git
   git init
   
   # Agregar todos los archivos
   git add .
   
   # Hacer el primer commit
   git commit -m "Initial commit - Sistema de predicción médica"
   
   # Conectar con tu repositorio (reemplaza TU_USUARIO y TU_REPO)
   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
   
   # Subir los archivos
   git branch -M main
   git push -u origin main
   ```

### Paso 2: Deploy en Render

1. **Crear cuenta en Render:** https://render.com (usa tu cuenta de GitHub)

2. **Crear nuevo Web Service:**
   - Click en "New +" → "Web Service"
   - Conectar tu repositorio de GitHub
   - Seleccionar el repositorio del proyecto

3. **Configuración del servicio:**
   - **Name:** `prediccion-medica` (o el que prefieras)
   - **Region:** Oregon (USA) o la más cercana
   - **Branch:** `main`
   - **Root Directory:** (dejar vacío)
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** `Free`

4. **Variables de entorno (opcional):**
   - No necesitas agregar ninguna por ahora

5. **Click en "Create Web Service"**
   - Espera 5-10 minutos mientras se construye
   - Render instalará todas las dependencias automáticamente
   - Te dará una URL como: `https://prediccion-medica.onrender.com`

6. **¡Listo!** Tu app estará disponible en la URL proporcionada

### Notas importantes sobre Render (Plan Gratuito):
- ⚠️ La app se "duerme" después de 15 min de inactividad
- ⏱️ Primera carga después de dormir toma ~30-50 segundos
- 🔄 Se activa automáticamente cuando alguien accede
- 📊 750 horas gratis al mes (suficiente para uso académico)

---

## Opción 2: PythonAnywhere (GRATIS)

### Paso 1: Crear cuenta
1. Ir a: https://www.pythonanywhere.com
2. Crear cuenta gratuita (Beginner account)

### Paso 2: Subir archivos
1. En Dashboard → Files → Upload files
2. Subir todos los archivos del proyecto (puede tomar tiempo por los .pkl)

### Paso 3: Configurar Web App
1. Web → Add a new web app
2. Python 3.10
3. Flask
4. Configurar paths

### Paso 4: Instalar dependencias
```bash
pip install --user -r requirements.txt
```

### URL final:
`https://TU_USUARIO.pythonanywhere.com`

---

## Opción 3: Railway.app (GRATIS con límites)

### Paso 1: Crear cuenta
1. Ir a: https://railway.app
2. Login con GitHub

### Paso 2: Deploy
1. "New Project" → "Deploy from GitHub repo"
2. Seleccionar tu repositorio
3. Railway detecta Flask automáticamente
4. Deploy automático

### URL final:
Railway genera una URL automática

---

## ⚡ Deploy Rápido sin GitHub (Render desde CLI)

Si no quieres usar GitHub, puedes usar Render CLI:

```bash
# Instalar Render CLI
npm install -g render-cli

# Login
render login

# Deploy
render deploy
```

---

## 📋 Checklist antes de Deploy

✅ Archivos creados:
- [x] `Procfile` - Comando para iniciar la app
- [x] `requirements.txt` - Dependencias de Python actualizado
- [x] `runtime.txt` - Versión de Python
- [x] `.gitignore` - Archivos a ignorar en Git
- [x] `uploads/.gitkeep` - Mantener carpeta uploads

✅ Código actualizado:
- [x] `app.py` configurado para puerto dinámico
- [x] Debug=False en producción
- [x] Host='0.0.0.0' para acceso público

✅ Archivos importantes incluidos:
- [x] Todos los archivos `.pkl` (modelos)
- [x] `DEMALE-HSJM_2025_data.xlsx` (dataset)
- [x] Carpetas `static/` y `templates/`

---

## 🔧 Solución de Problemas

### Error: "Application failed to start"
- Verificar que `requirements.txt` esté completo
- Revisar logs en Render Dashboard

### Error: "Module not found"
- Agregar el módulo faltante a `requirements.txt`
- Hacer commit y push nuevamente

### App muy lenta
- Normal en plan gratuito después de inactividad
- Considera upgrade si necesitas velocidad constante

### Archivos .pkl muy grandes
- Los modelos suman ~300KB (está bien)
- Si hay problemas, considera usar Git LFS

---

## 💡 Recomendación Final

**Para proyecto académico/presentación:**
👉 **RENDER.COM** es la mejor opción:
- ✅ Gratis
- ✅ Fácil de configurar
- ✅ URL profesional con HTTPS
- ✅ Deploy automático desde GitHub
- ✅ Logs y monitoring incluidos

**Tiempo estimado de setup:** 15-20 minutos

---

## 📱 Compartir la App

Una vez deployada, solo comparte la URL:
- Render: `https://tu-app.onrender.com`
- PythonAnywhere: `https://tu-usuario.pythonanywhere.com`
- Railway: URL generada automáticamente

¡Cualquier persona con el link podrá acceder! 🎉
