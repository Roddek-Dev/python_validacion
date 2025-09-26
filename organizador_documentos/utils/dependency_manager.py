"""
Gestor de dependencias opcionales del sistema.
"""

import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class DependencyManager:
    """Gestiona las dependencias opcionales y capabilities del sistema."""
    
    def __init__(self):
        self.capabilities = self._detect_capabilities()
        self._log_capabilities()
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detecta qué capabilities están disponibles basadas en dependencias."""
        capabilities = {}
        
        # PDF processing
        try:
            import pdfplumber
            capabilities['pdf_processing'] = True
            logger.debug("pdfplumber disponible")
        except ImportError:
            capabilities['pdf_processing'] = False
            logger.warning("pdfplumber no disponible")
        
        # DOCX processing
        try:
            from docx import Document
            capabilities['docx_processing'] = True
            logger.debug("python-docx disponible")
        except ImportError:
            capabilities['docx_processing'] = False
            logger.warning("python-docx no disponible")
        
        # XLSX processing
        try:
            from openpyxl import load_workbook
            capabilities['xlsx_processing'] = True
            logger.debug("openpyxl disponible")
        except ImportError:
            capabilities['xlsx_processing'] = False
            logger.warning("openpyxl no disponible")
        
        # OCR processing
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import pytesseract
            capabilities['ocr_processing'] = True
            logger.debug("OCR (Pillow + pytesseract) disponible")
        except ImportError:
            capabilities['ocr_processing'] = False
            logger.warning("OCR no disponible")
        
        # PDF to image conversion
        try:
            from pdf2image import convert_from_path
            capabilities['pdf_to_image'] = True
            logger.debug("pdf2image disponible")
        except ImportError:
            capabilities['pdf_to_image'] = False
            logger.warning("pdf2image no disponible")
        
        return capabilities
    
    def _log_capabilities(self) -> None:
        """Registra las capabilities detectadas."""
        enabled = [cap for cap, available in self.capabilities.items() if available]
        disabled = [cap for cap, available in self.capabilities.items() if not available]
        
        if enabled:
            logger.info(f"Capabilities habilitadas: {', '.join(enabled)}")
        if disabled:
            logger.info(f"Capabilities deshabilitadas: {', '.join(disabled)}")
    
    def check_required_dependencies(self, config: Dict) -> Dict[str, str]:
        """
        Verifica que las dependencias requeridas estén disponibles según la configuración.
        
        Args:
            config: Configuración del sistema
            
        Returns:
            Dict con dependencias faltantes y razones
        """
        missing = {}
        
        # Verificar dependencias para análisis de contenido
        if config.get("enable_content_search", False):
            content_deps = {
                'pdf_processing': 'pdfplumber',
                'docx_processing': 'python-docx', 
                'xlsx_processing': 'openpyxl'
            }
            
            for cap, dep_name in content_deps.items():
                if not self.capabilities.get(cap, False):
                    missing[dep_name] = "Requerido para análisis de contenido"
        
        # Verificar dependencias para OCR
        if config.get("enable_ocr", False):
            if not self.capabilities.get('ocr_processing', False):
                missing['Pillow + pytesseract'] = "Requerido para OCR"
                missing['tesseract-system'] = "Instale tesseract en su sistema"
        
        return missing
    
    def get_available_extractors(self) -> List[str]:
        """Retorna lista de extractores de contenido disponibles."""
        extractors = []
        
        if self.capabilities.get('pdf_processing'):
            extractors.append('pdf')
        if self.capabilities.get('docx_processing'):
            extractors.append('docx')
        if self.capabilities.get('xlsx_processing'):
            extractors.append('xlsx')
        
        extractors.append('txt')  # Siempre disponible
        
        return extractors
    
    def can_process_ocr(self) -> bool:
        """Verifica si el procesamiento OCR está disponible."""
        return self.capabilities.get('ocr_processing', False)
    
    def can_convert_pdf_to_image(self) -> bool:
        """Verifica si la conversión PDF a imagen está disponible."""
        return self.capabilities.get('pdf_to_image', False)
    
    def get_missing_dependencies_message(self, missing_deps: Dict[str, str]) -> str:
        """Genera mensaje informativo sobre dependencias faltantes."""
        if not missing_deps:
            return ""
        
        message = "Para habilitar todas las funcionalidades, instale:\n"
        
        # Agrupar por tipo de instalación
        pip_deps = []
        system_deps = []
        
        for dep, reason in missing_deps.items():
            if 'system' in dep.lower() or 'tesseract' in dep.lower():
                system_deps.append(f"  - {dep}: {reason}")
            else:
                pip_deps.append(f"  - pip install {dep}")
        
        if pip_deps:
            message += "\nDependencias Python:\n" + "\n".join(pip_deps)
        
        if system_deps:
            message += "\nDependencias del sistema:\n" + "\n".join(system_deps)
        
        return message