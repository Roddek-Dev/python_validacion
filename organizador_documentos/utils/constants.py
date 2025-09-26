"""
Constantes del sistema de organización de documentos.
"""

from typing import Dict, List, Set

# Extensiones de archivo soportadas
SUPPORTED_EXTENSIONS: Set[str] = {
    ".pdf", ".docx", ".xlsx", ".jpg", ".jpeg", ".png", ".txt", ".tiff", ".bmp"
}

# Extensiones para análisis de contenido
CONTENT_ANALYSIS_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".xlsx"]

# Extensiones de imagen para OCR
IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}

# Configuración por defecto
DEFAULT_CONFIG: Dict = {
    "dry_run": True,
    "default_mode": "copy",
    "num_threads": 4,
    "confidence_threshold": 15,
    "enable_content_search": False,
    "pdf_pages_to_read": 1,
    "extensions_for_content_search": CONTENT_ANALYSIS_EXTENSIONS,
    "enable_ocr": False,
    "ocr_languages": "spa",
    "ocr_max_pages": 1,
    "scanned_pdf_threshold": 50,
    "ocr_min_text_threshold": 30,
    "max_filename_length": 100,
    "enable_user_organization": False,
    "verbose": False,
    "log_file": "proceso.log",
    "csv_file": "resultados.csv",
}

# Patrones para detectar carpetas de usuario
USER_PATTERNS: List[str] = [
    r"([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑ\s]+)\s+-\s+CC\s+\d+",  # "NOMBRE APELLIDO - CC 123456789"
    r"([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑ\s]+)\s+CC\s+\d+",      # "NOMBRE APELLIDO CC 123456789"
    r"([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑ\s]+)\s+-\s+\d+",       # "NOMBRE APELLIDO - 123456789"
]

# Keywords genéricas que reciben puntuación baja
GENERIC_KEYWORDS: List[str] = [
    "certificado", "certificacion", "formato", "documento", 
    "ingreso", "laboral", "personal"
]

# Keywords críticas específicas por categoría
CRITICAL_SPECIFIC_KEYWORDS: Dict[str, List[str]] = {
    "06": [
        "afiliacion en nuestra arl", 
        "afiliacion en el ramo de resgos laborales", 
        "afiliacion arl", 
        "certificado de afiliacion"
    ],
    "10": [
        "documento de identidad", 
        "republica de colombia", 
        "cedula de ciudadania"
    ]
}

# Caracteres inválidos para nombres de archivo
INVALID_FILENAME_CHARS: str = '<>:"/\\|?*'

# Configuración de OCR
OCR_CONFIG_TEMPLATE: str = "--oem 3 --psm 6 -l {language}"

# Configuración de cache
CACHE_CONFIG: Dict = {
    "max_size": 100,
    "ttl_seconds": 3600,  # 1 hora
    "min_file_size_for_cache": 1_000_000,  # 1MB
}

# Configuración de procesamiento por lotes
BATCH_CONFIG: Dict = {
    "small_file_threshold": 100_000,  # 100KB
    "batch_size": 50,
    "io_thread_multiplier": 2,
}

# Mensajes de razones de clasificación
CLASSIFICATION_REASONS: Dict[str, str] = {
    "baja_confianza": "Baja confianza (puntuación insuficiente)",
    "sin_coincidencias": "Sin coincidencias (no se encontraron keywords)",
    "empate_post_penalizacion": "Empate después de penalizaciones",
    "sin_clasificar": "Sin clasificar (puntuación baja)",
    "ambiguo": "Ambiguo (empate en categorías)",
    "carpeta_padre": "Clasificado por carpeta padre",
    "contenido": "Clasificado por análisis de contenido",
    "ocr": "Clasificado por OCR",
    "nombre_archivo": "Clasificado por nombre de archivo",
}