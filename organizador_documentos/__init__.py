"""
Organizador de Documentos - Sistema de clasificación automática de documentos.

Un sistema modular para clasificar, organizar y renombrar documentos usando:
- Algoritmos de puntuación basados en palabras clave
- Análisis de contenido de múltiples formatos
- OCR para imágenes y PDFs escaneados
- Procesamiento paralelo optimizado
"""

__version__ = "2.0.0"
__author__ = "Document Organizer Team"

from .core.document_organizer import DocumentOrganizer
from .core.config_manager import ConfigManager

__all__ = ["DocumentOrganizer", "ConfigManager"]