"""
Procesador OCR para imágenes y PDFs escaneados.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple
import hashlib
import time

from ..utils.constants import IMAGE_EXTENSIONS, OCR_CONFIG_TEMPLATE, CACHE_CONFIG

logger = logging.getLogger(__name__)


class OCRCache:
    """Cache LRU para resultados de OCR."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _is_expired(self, key: str) -> bool:
        """Verifica si una entrada del cache ha expirado."""
        if key not in self.access_times:
            return True
        return time.time() - self.access_times[key] > self.ttl_seconds
    
    def _cleanup_expired(self) -> None:
        """Limpia entradas expiradas del cache."""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
    
    def get(self, key: str) -> Optional[str]:
        """Obtiene valor del cache."""
        if key in self.cache and not self._is_expired(key):
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, value: str) -> None:
        """Almacena valor en cache."""
        self._cleanup_expired()
        
        # Si el cache está lleno, remover el elemento menos usado
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self.cache.pop(oldest_key, None)
            self.access_times.pop(oldest_key, None)
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def get_stats(self) -> Dict[str, int]:
        """Retorna estadísticas del cache."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": 0  # Se calculará externamente
        }


class OCRProcessor:
    """Procesador OCR para imágenes y PDFs escaneados."""
    
    def __init__(self, capabilities: Dict[str, bool], config: Dict):
        self.capabilities = capabilities
        self.config = config
        self.cache = OCRCache(
            max_size=CACHE_CONFIG["max_size"],
            ttl_seconds=CACHE_CONFIG["ttl_seconds"]
        )
        self.stats = {
            "ocr_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "processing_time": 0.0
        }
        
        # Verificar disponibilidad de OCR
        if not self.capabilities.get('ocr_processing'):
            logger.warning("OCR no disponible - instale Pillow y pytesseract")
    
    def can_process_ocr(self) -> bool:
        """Verifica si el procesamiento OCR está disponible."""
        return self.capabilities.get('ocr_processing', False)
    
    def extract_text_with_ocr(self, file_path: Path) -> str:
        """
        Extrae texto usando OCR con cache inteligente.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Texto extraído por OCR
        """
        if not self.can_process_ocr():
            return ""
        
        start_time = time.time()
        
        try:
            # Generar clave de cache
            cache_key = self._generate_cache_key(file_path)
            
            # Verificar cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit para OCR: {file_path.name}")
                return cached_result
            
            self.stats["cache_misses"] += 1
            
            # Procesar según tipo de archivo
            extension = file_path.suffix.lower()
            
            if extension == ".pdf":
                text = self._process_pdf_ocr(file_path)
            elif extension in IMAGE_EXTENSIONS:
                text = self._process_image_ocr(file_path)
            else:
                logger.warning(f"Tipo de archivo no soportado para OCR: {extension}")
                return ""
            
            # Almacenar en cache si vale la pena
            if self._should_cache_result(file_path, text):
                self.cache.put(cache_key, text)
            
            self.stats["ocr_calls"] += 1
            self.stats["processing_time"] += time.time() - start_time
            
            if text.strip():
                logger.debug(f"OCR exitoso: {file_path.name} ({len(text)} chars)")
            
            return text
            
        except Exception as e:
            logger.error(f"Error en OCR para {file_path}: {e}")
            return ""
    
    def _generate_cache_key(self, file_path: Path) -> str:
        """Genera clave única para cache basada en archivo."""
        try:
            file_stat = file_path.stat()
            key_data = f"{file_path.name}_{file_stat.st_size}_{file_stat.st_mtime}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception:
            # Fallback a hash del nombre del archivo
            return hashlib.md5(str(file_path).encode()).hexdigest()
    
    def _should_cache_result(self, file_path: Path, text: str) -> bool:
        """Determina si vale la pena cachear el resultado."""
        try:
            file_size = file_path.stat().st_size
            min_size = CACHE_CONFIG["min_file_size_for_cache"]
            min_text_length = self.config.get("ocr_min_text_threshold", 30)
            
            return (file_size >= min_size and 
                    len(text.strip()) >= min_text_length)
        except Exception:
            return False
    
    def _process_pdf_ocr(self, file_path: Path) -> str:
        """Procesa PDF con OCR."""
        # Primero verificar si es PDF escaneado
        if not self._is_pdf_scanned(file_path):
            # Si no es escaneado, intentar extracción normal
            return self._extract_pdf_text_normal(file_path)
        
        # Procesar como PDF escaneado
        if not self.capabilities.get('pdf_to_image'):
            logger.warning("pdf2image no disponible para OCR en PDFs")
            return ""
        
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            # Convertir solo la primera página
            pages = convert_from_path(
                str(file_path), 
                dpi=300, 
                first_page=1, 
                last_page=self.config.get("ocr_max_pages", 1)
            )
            
            text_parts = []
            ocr_config = OCR_CONFIG_TEMPLATE.format(
                language=self.config.get("ocr_languages", "spa")
            )
            
            for page_img in pages:
                processed_img = self._preprocess_image_for_ocr(page_img)
                page_text = pytesseract.image_to_string(processed_img, config=ocr_config)
                if page_text.strip():
                    text_parts.append(page_text)
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error en OCR de PDF {file_path}: {e}")
            return ""
    
    def _process_image_ocr(self, file_path: Path) -> str:
        """Procesa imagen con OCR."""
        try:
            from PIL import Image
            import pytesseract
            
            image = Image.open(file_path)
            processed_image = self._preprocess_image_for_ocr(image)
            
            ocr_config = OCR_CONFIG_TEMPLATE.format(
                language=self.config.get("ocr_languages", "spa")
            )
            
            text = pytesseract.image_to_string(processed_image, config=ocr_config)
            return text
            
        except Exception as e:
            logger.error(f"Error en OCR de imagen {file_path}: {e}")
            return ""
    
    def _preprocess_image_for_ocr(self, image) -> object:
        """Pre-procesa imagen para mejorar OCR."""
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            
            # Convertir a escala de grises
            if image.mode != "L":
                image = image.convert("L")

            # Mejorar contraste
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Aplicar filtro de nitidez
            image = image.filter(ImageFilter.SHARPEN)

            # Redimensionar si es muy pequeña
            width, height = image.size
            if width < 1200 or height < 1200:
                scale_factor = max(1200 / width, 1200 / height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            return image
            
        except Exception as e:
            logger.error(f"Error pre-procesando imagen: {e}")
            return image
    
    def _is_pdf_scanned(self, file_path: Path) -> bool:
        """Detecta si un PDF es escaneado y necesita OCR."""
        if not self.capabilities.get('pdf_processing'):
            return True

        try:
            import pdfplumber
            
            text_content = ""
            with pdfplumber.open(file_path) as pdf:
                # Analizar solo las primeras 2 páginas
                for i, page in enumerate(pdf.pages[:2]):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text

            text_len = len(text_content.strip())
            threshold = self.config.get("scanned_pdf_threshold", 100)
            
            if text_len < threshold:
                return True

            # Verificar ratio de caracteres extraños
            weird_chars = sum(1 for c in text_content if ord(c) > 127 or c in "□■▪▫")
            weird_ratio = weird_chars / max(text_len, 1)
            if weird_ratio > 0.3:
                return True

            # Verificar ratio de espacios (texto mal extraído)
            space_ratio = text_content.count(" ") / max(text_len, 1)
            if space_ratio > 0.7:
                return True
                
            return False
            
        except Exception as e:
            logger.warning(f"Error verificando PDF escaneado {file_path}: {e}")
            return True
    
    def _extract_pdf_text_normal(self, file_path: Path) -> str:
        """Extrae texto normal de PDF (no escaneado)."""
        if not self.capabilities.get('pdf_processing'):
            return ""
        
        try:
            import pdfplumber
            
            text = ""
            max_pages = self.config.get("pdf_pages_to_read", 1)
            
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages[:max_pages]):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as page_error:
                        logger.warning(f"Error procesando página {i+1} de {file_path}: {page_error}")
                        continue
            
            return text
            
        except Exception as e:
            logger.error(f"Error extrayendo texto normal de PDF {file_path}: {e}")
            return ""
    
    def get_stats(self) -> Dict[str, any]:
        """Retorna estadísticas del procesador OCR."""
        cache_stats = self.cache.get_stats()
        
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = (self.stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            "cache_hit_rate": hit_rate,
            "cache_size": cache_stats["size"],
            "avg_processing_time": (
                self.stats["processing_time"] / max(self.stats["ocr_calls"], 1)
            )
        }
    
    def clear_cache(self) -> None:
        """Limpia el cache de OCR."""
        self.cache.cache.clear()
        self.cache.access_times.clear()
        logger.info("Cache de OCR limpiado")