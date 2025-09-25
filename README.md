# Organizador de Documentos

Script Python para automatizar la clasificación y organización de documentos personales usando un algoritmo de puntuación inteligente.

## Instalación de Dependencias

\`\`\`bash
pip install PyYAML pdfplumber python-docx openpyxl Pillow pytesseract tqdm pdf2image
\`\`\`

Para OCR, también necesita instalar Tesseract:
- Windows: Descargar desde https://github.com/UB-Mannheim/tesseract/wiki
- macOS: `brew install tesseract`
- Linux (Arch): `sudo pacman -S tesseract tesseract-data`
- Para OCR en PDFs escaneados: necesita Poppler (proporciona `pdftoppm`)
  - Arch: `sudo pacman -S poppler`

## Uso Básico

\`\`\`bash
# Simulación (dry-run por defecto)
python organizador_documentos.py --origen /ruta/documentos --destino /ruta/organizada

# Ejecución real
python organizador_documentos.py --origen /ruta/documentos --destino /ruta/organizada --dry-run false

# Con análisis de contenido y OCR (incluye OCR para PDF escaneado si tiene pdf2image + Poppler)
python organizador_documentos.py --origen /ruta/documentos --destino /ruta/organizada --enable-content --enable-ocr --dry-run false
\`\`\`

## Características

- **Clasificación inteligente**: Algoritmo de puntuación basado en prefijos numéricos, palabras clave, contenido y OCR
- **42+ categorías**: Predefinidas para documentos de recursos humanos
- **Detección de duplicados**: Por hash SHA256
- **Renombrado automático**: Formato estándar con prevención de conflictos
- **Procesamiento concurrente**: Múltiples hilos para mejor rendimiento
- **Logging completo**: Archivo de log y reporte CSV detallado
- **Modo seguro**: Dry-run por defecto para probar antes de ejecutar

## Configuración

El archivo `config.yml` permite personalizar:
- Categorías y palabras clave
- Configuración de OCR y análisis de contenido
- Reglas de renombrado
- Parámetros de procesamiento

## Salidas

- **Carpetas organizadas**: Estructura numerada según categorías
- **resultados.csv**: Reporte detallado de cada archivo procesado
- **proceso.log**: Log completo de la ejecución
- **Resumen en consola**: Estadísticas finales del procesamiento
