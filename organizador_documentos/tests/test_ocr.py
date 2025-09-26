"""
Tests para el procesador OCR.
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from organizador_documentos.core.ocr_processor import OCRProcessor, OCRCache


class TestOCRCache(unittest.TestCase):
    """Tests para OCRCache."""
    
    def setUp(self):
        """Configuración inicial para tests."""
        self.cache = OCRCache(max_size=3, ttl_seconds=1)
    
    def test_cache_put_and_get(self):
        """Test de almacenamiento y recuperación del cache."""
        self.cache.put("key1", "value1")
        result = self.cache.get("key1")
        
        self.assertEqual(result, "value1")
    
    def test_cache_miss(self):
        """Test de cache miss."""
        result = self.cache.get("nonexistent_key")
        self.assertIsNone(result)
    
    def test_cache_expiration(self):
        """Test de expiración del cache."""
        self.cache.put("key1", "value1")
        
        # Verificar que está disponible
        self.assertEqual(self.cache.get("key1"), "value1")
        
        # Esperar a que expire
        time.sleep(1.1)
        
        # Verificar que expiró
        result = self.cache.get("key1")
        self.assertIsNone(result)
    
    def test_cache_size_limit(self):
        """Test de límite de tamaño del cache."""
        # Llenar el cache hasta el límite
        for i in range(4):  # max_size es 3
            self.cache.put(f"key{i}", f"value{i}")
        
        # El primer elemento debería haber sido removido
        self.assertIsNone(self.cache.get("key0"))
        self.assertEqual(self.cache.get("key3"), "value3")
    
    def test_cache_stats(self):
        """Test de estadísticas del cache."""
        stats = self.cache.get_stats()
        
        expected_keys = ["size", "max_size", "hit_rate"]
        for key in expected_keys:
            self.assertIn(key, stats)


class TestOCRProcessor(unittest.TestCase):
    """Tests para OCRProcessor."""
    
    def setUp(self):
        """Configuración inicial para tests."""
        self.capabilities = {
            'ocr_processing': True,
            'pdf_processing': True,
            'pdf_to_image': True,
        }
        self.config = {
            'ocr_languages': 'spa',
            'ocr_max_pages': 1,
            'scanned_pdf_threshold': 50,
            'ocr_min_text_threshold': 30,
        }
        
        self.processor = OCRProcessor(self.capabilities, self.config)
    
    def test_initialization_without_ocr_capability(self):
        """Test de inicialización sin capability de OCR."""
        no_ocr_caps = {'ocr_processing': False}
        processor = OCRProcessor(no_ocr_caps, self.config)
        
        self.assertFalse(processor.can_process_ocr())
    
    def test_can_process_ocr(self):
        """Test de verificación de capacidad de OCR."""
        self.assertTrue(self.processor.can_process_ocr())
        
        # Sin capability
        processor = OCRProcessor({'ocr_processing': False}, self.config)
        self.assertFalse(processor.can_process_ocr())
    
    def test_generate_cache_key(self):
        """Test de generación de clave de cache."""
        # Mock del archivo
        mock_path = Mock(spec=Path)
        mock_path.name = "test.pdf"
        mock_stat = Mock()
        mock_stat.st_size = 1000
        mock_stat.st_mtime = 1234567890
        mock_path.stat.return_value = mock_stat
        
        key = self.processor._generate_cache_key(mock_path)
        
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 32)  # MD5 hash length
    
    def test_should_cache_result(self):
        """Test de decisión de cacheo."""
        # Mock del archivo grande
        mock_path = Mock(spec=Path)
        mock_stat = Mock()
        mock_stat.st_size = 2_000_000  # 2MB
        mock_path.stat.return_value = mock_stat
        
        # Texto suficientemente largo
        long_text = "a" * 50
        
        result = self.processor._should_cache_result(mock_path, long_text)
        self.assertTrue(result)
        
        # Archivo pequeño
        mock_stat.st_size = 1000  # 1KB
        result = self.processor._should_cache_result(mock_path, long_text)
        self.assertFalse(result)
        
        # Texto muy corto
        mock_stat.st_size = 2_000_000  # 2MB
        short_text = "abc"
        result = self.processor._should_cache_result(mock_path, short_text)
        self.assertFalse(result)
    
    @patch('organizador_documentos.core.ocr_processor.pdfplumber')
    def test_is_pdf_scanned_normal_pdf(self, mock_pdfplumber):
        """Test de detección de PDF normal (no escaneado)."""
        # Mock de PDF con texto suficiente
        mock_page = Mock()
        mock_page.extract_text.return_value = "Este es un texto normal con suficiente contenido para no ser considerado escaneado"
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        file_path = Path("documento.pdf")
        result = self.processor._is_pdf_scanned(file_path)
        
        self.assertFalse(result)
    
    @patch('organizador_documentos.core.ocr_processor.pdfplumber')
    def test_is_pdf_scanned_scanned_pdf(self, mock_pdfplumber):
        """Test de detección de PDF escaneado."""
        # Mock de PDF con poco texto
        mock_page = Mock()
        mock_page.extract_text.return_value = "abc"  # Muy poco texto
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        file_path = Path("documento.pdf")
        result = self.processor._is_pdf_scanned(file_path)
        
        self.assertTrue(result)
    
    @patch('organizador_documentos.core.ocr_processor.pytesseract')
    @patch('organizador_documentos.core.ocr_processor.Image')
    def test_process_image_ocr(self, mock_image_class, mock_pytesseract):
        """Test de procesamiento OCR de imagen."""
        # Mock de la imagen
        mock_image = Mock()
        mock_image_class.open.return_value = mock_image
        
        # Mock del resultado OCR
        mock_pytesseract.image_to_string.return_value = "Texto extraído por OCR"
        
        # Mock del preprocesamiento
        with patch.object(self.processor, '_preprocess_image_for_ocr', return_value=mock_image):
            file_path = Path("imagen.jpg")
            result = self.processor._process_image_ocr(file_path)
        
        self.assertEqual(result, "Texto extraído por OCR")
        mock_image_class.open.assert_called_once_with(file_path)
        mock_pytesseract.image_to_string.assert_called_once()
    
    @patch('organizador_documentos.core.ocr_processor.convert_from_path')
    @patch('organizador_documentos.core.ocr_processor.pytesseract')
    def test_process_pdf_ocr(self, mock_pytesseract, mock_convert):
        """Test de procesamiento OCR de PDF."""
        # Mock de conversión PDF a imagen
        mock_image = Mock()
        mock_convert.return_value = [mock_image]
        
        # Mock del resultado OCR
        mock_pytesseract.image_to_string.return_value = "Texto de PDF escaneado"
        
        # Mock de verificación de PDF escaneado
        with patch.object(self.processor, '_is_pdf_scanned', return_value=True):
            with patch.object(self.processor, '_preprocess_image_for_ocr', return_value=mock_image):
                file_path = Path("documento.pdf")
                result = self.processor._process_pdf_ocr(file_path)
        
        self.assertEqual(result, "Texto de PDF escaneado")
        mock_convert.assert_called_once()
        mock_pytesseract.image_to_string.assert_called_once()
    
    def test_extract_text_with_ocr_no_capability(self):
        """Test de OCR sin capability disponible."""
        processor = OCRProcessor({'ocr_processing': False}, self.config)
        
        file_path = Path("imagen.jpg")
        result = processor.extract_text_with_ocr(file_path)
        
        self.assertEqual(result, "")
    
    def test_extract_text_with_ocr_cache_hit(self):
        """Test de OCR con cache hit."""
        file_path = Path("imagen.jpg")
        
        # Simular cache hit
        with patch.object(self.processor.cache, 'get', return_value="Texto cacheado"):
            result = self.processor.extract_text_with_ocr(file_path)
        
        self.assertEqual(result, "Texto cacheado")
        self.assertEqual(self.processor.stats["cache_hits"], 1)
    
    @patch('organizador_documentos.core.ocr_processor.ImageEnhance')
    @patch('organizador_documentos.core.ocr_processor.ImageFilter')
    def test_preprocess_image_for_ocr(self, mock_filter, mock_enhance):
        """Test de preprocesamiento de imagen."""
        # Mock de la imagen
        mock_image = Mock()
        mock_image.mode = "RGB"
        mock_image.convert.return_value = mock_image
        mock_image.size = (800, 600)  # Imagen pequeña que necesita redimensionamiento
        mock_image.resize.return_value = mock_image
        mock_image.filter.return_value = mock_image
        
        # Mock del enhancer
        mock_enhancer = Mock()
        mock_enhancer.enhance.return_value = mock_image
        mock_enhance.Contrast.return_value = mock_enhancer
        
        result = self.processor._preprocess_image_for_ocr(mock_image)
        
        # Verificar que se aplicaron las transformaciones
        mock_image.convert.assert_called_with("L")  # Conversión a escala de grises
        mock_enhancer.enhance.assert_called_with(2.0)  # Mejora de contraste
        mock_image.resize.assert_called_once()  # Redimensionamiento
    
    def test_get_stats(self):
        """Test de obtención de estadísticas."""
        # Simular algunas operaciones
        self.processor.stats["ocr_calls"] = 5
        self.processor.stats["cache_hits"] = 2
        self.processor.stats["cache_misses"] = 3
        self.processor.stats["processing_time"] = 10.0
        
        stats = self.processor.get_stats()
        
        expected_keys = [
            "ocr_calls", "cache_hits", "cache_misses", "processing_time",
            "cache_hit_rate", "cache_size", "avg_processing_time"
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Verificar cálculos
        self.assertEqual(stats["cache_hit_rate"], 40.0)  # 2/5 * 100
        self.assertEqual(stats["avg_processing_time"], 2.0)  # 10.0/5
    
    def test_clear_cache(self):
        """Test de limpieza del cache."""
        # Añadir algo al cache
        self.processor.cache.put("key1", "value1")
        self.assertIsNotNone(self.processor.cache.get("key1"))
        
        # Limpiar cache
        self.processor.clear_cache()
        
        # Verificar que se limpió
        self.assertIsNone(self.processor.cache.get("key1"))
        self.assertEqual(len(self.processor.cache.cache), 0)


if __name__ == '__main__':
    unittest.main()