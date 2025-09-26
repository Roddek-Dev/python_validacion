# Organizador de Documentos

Script de automatización para clasificación, organización y renombrado de documentos personales usando algoritmo de puntuación basado en palabras clave, análisis de contenido y OCR.

## Características

- **Clasificación Inteligente**: Algoritmo de puntuación jerárquico basado en múltiples factores
- **Análisis de Contenido**: Extracción de texto de PDFs, DOCX, XLSX y TXT
- **OCR Avanzado**: Reconocimiento óptico de caracteres para imágenes y PDFs escaneados
- **Procesamiento Paralelo**: Utiliza múltiples hilos para mayor velocidad
- **Detección de Duplicados**: Identifica archivos duplicados usando hash SHA256
- **Renombrado Inteligente**: Nombres de archivo organizados por categoría
- **Configuración Flexible**: Archivo YAML para personalizar comportamiento

## Instalación

### Dependencias del Sistema (Arch Linux)

```bash
sudo pacman -S python python-pip tesseract tesseract-data poppler
```

### Dependencias de Python

```bash
pip install PyYAML pdfplumber python-docx openpyxl Pillow pytesseract tqdm pdf2image
```

## Uso

### Comando Básico

```bash
python organizador_documentos.py --enable-ocr --enable-content --verbose
```

### Comandos de Ejecución

```bash
# Simulación (recomendado para probar)
python organizador_documentos.py --enable-ocr --enable-content --verbose

# Ejecución real con organización por usuario
python organizador_documentos.py --enable-ocr --enable-content --enable-users --verbose --dry-run false

# Ejecución real sin organización por usuario (estructura tradicional)
python organizador_documentos.py --enable-ocr --enable-content --verbose --dry-run false

# Con configuración personalizada
python organizador_documentos.py --config mi_config.yml --enable-ocr --enable-content --verbose

# Con número específico de hilos
python organizador_documentos.py --threads 8 --enable-ocr --enable-content --enable-users --verbose
```

### Argumentos Disponibles

- `--origen`: Ruta a carpeta origen (por defecto: ~/Escritorio/Documentos_Desordenados)
- `--destino`: Ruta a carpeta destino (por defecto: ~/Escritorio/Documentos_Organizados)
- `--config`: Archivo de configuración personalizado
- `--threads`: Número de hilos para procesamiento (sobrescribe config.yml)
- `--enable-content`: Habilita análisis de contenido
- `--enable-ocr`: Habilita OCR para imágenes y PDFs escaneados
- `--enable-users`: Habilita organización por usuario
- `--dry-run`: Simula proceso sin mover archivos (true/false)
- `--verbose`: Activa logs detallados

## Configuración

El archivo `config.yml` permite personalizar el comportamiento del script:

### Configuración de Rendimiento

```yaml
num_threads: 6                    # Número de hilos (recomendado: 6-8 para tu PC)
confidence_threshold: 15          # Umbral de confianza para clasificación
pdf_pages_to_read: 1             # Páginas a leer de PDFs (1 = más rápido)
ocr_max_pages: 1                 # Páginas para OCR (1 = más rápido)
```

### Configuración de Categorías

Las categorías y keywords se configuran en `config.yml`. Cada categoría puede tener máximo 5 keywords para optimizar rendimiento.


### Organización por Usuario

```yaml
enable_user_organization: true  # Habilita organización por usuario
```

**Funcionamiento:**
- Detecta automáticamente el nombre del usuario de la estructura de carpetas origen
- Crea carpetas de usuario en la raíz del destino
- Estructura: `Documentos_Organizados/Usuario/Categoria/Archivo`
- Ejemplo: `Documentos_Organizados/Bautista_Gonzalez_Carlos_Andres/06 Afiliación ARL/archivo.pdf`


## Algoritmo de Clasificación

1. **Análisis de Nombre de Archivo**: Búsqueda de keywords en el nombre
2. **Análisis de Carpeta Padre**: Keywords en la carpeta contenedora
3. **Análisis de Contenido**: Extracción de texto de archivos (opcional)
4. **OCR**: Reconocimiento óptico para imágenes y PDFs escaneados (opcional)
5. **Sistema de Puntuación**: Asigna puntos según especificidad de keywords
6. **Penalizaciones**: Reduce puntuación por ambigüedad entre categorías

## Salidas

### Archivos Generados

- `resultados.csv`: Detalles de clasificación de cada archivo
- `proceso.log`: Log detallado del procesamiento
- Carpetas organizadas por categoría en destino

### Resumen de Procesamiento

- Total de archivos procesados
- Archivos clasificados por categoría
- Archivos pendientes de revisar
- Archivos duplicados encontrados
- Estadísticas de uso de OCR y análisis de contenido
- Carpetas de usuario creadas

## Optimizaciones Implementadas

- **Reducción de Keywords**: Máximo 5 keywords por categoría
- **Procesamiento de PDFs**: Solo primera página para análisis y OCR
- **Configuración de Hilos**: Respeta configuración de `config.yml`
- **Código Optimizado**: Eliminación de redundancias y comentarios innecesarios
- **Gestión de Memoria**: Procesamiento eficiente de archivos grandes
- **Organización por Usuario**: Detección automática de usuarios y estructura jerárquica
- **Nombres Originales**: Mantiene los nombres originales de los archivos
- **Detección de Usuarios**: Patrones regex mejorados para caracteres especiales del español

## Solución de Problemas

### El script no respeta la configuración de hilos

**Problema**: Los argumentos de línea de comandos sobrescriben `config.yml`

**Solución**: Usar `--threads` solo si quieres sobrescribir, o no especificar para usar `config.yml`

### Rendimiento lento

**Soluciones**:
- Aumentar `num_threads` en `config.yml` (6-8 para tu PC)
- Reducir `pdf_pages_to_read` y `ocr_max_pages` a 1
- Verificar que no hay procesos que consuman CPU

### Archivos no clasificados

**Soluciones**:
- Reducir `confidence_threshold` (valor más bajo = más permisivo)
- Añadir keywords específicas en `config.yml`
- Revisar archivos en carpeta "Pendientes_Revisar"