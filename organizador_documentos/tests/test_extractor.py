"""
Tests para el extractor de contenido.
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from organizador_documentos.core.content_extractor import ContentExtractor


class TestContentExtractor(unittest.TestCase):
    """Tests para ContentExtractor."""
    
    def setUp(self):
        """Configuración inicial para tests."""
        self.capabilities = {
            'pdf_processing': True,
            'docx_processing': True,
            'xlsx_processing': True,
        }
        self.config = {
            'pdf_pages_to_read': 1,
            'xlsx_max_sheets': 3,
        }
        
        self.extractor = ContentExtractor(self.capabilities, self.config)
    
    def test_initialization_with_capabilities(self):
        """Test de inicialización con diferentes capabilities."""
        # Con todas las capabilities
        extractor = ContentExtractor(self.capabilities, self.config)
        self.assertIn('.pdf', extractor.extractors)
        self.assertIn('.docx', extractor.extractors)
        self.assertIn('.xlsx', extractor.extractors)
        self.assertIn('.txt', extractor.extractors)
        
        # Sin capabilities
        no_caps = {cap: False for cap in self.capabilities}
        extractor = ContentExtractor(no_caps, self.config)
        self.assertNotIn('.pdf', extractor.extractors)
        self.assertNotIn('.docx', extractor.extractors)
        self.assertNotIn('.xlsx', extractor.extractors)
        self.assertIn('.txt', extractor.extractors)  # TXT siempre disponible
    
    def test_can_extract(self):
        """Test de verificación de capacidad de extracción."""
        test_cases = [
            (Path("documento.pdf"), True),
            (Path("documento.docx"), True),
            (Path("documento.xlsx"), True),
            (Path("documento.txt"), True),
            (Path("imagen.jpg"), False),
            (Path("archivo.xyz"), False),
        ]
        
        for file_path, expected in test_cases:
            with self.subTest(file_path=file_path):
                result = self.extractor.can_extract(file_path)
                self.assertEqual(result, expected)
    
    @patch('builtins.open', new_callable=mock_open, read_data="Contenido del archivo TXT")
    def test_extract_from_txt(self, mock_file):
        """Test de extracción de archivo TXT."""
        file_path = Path("documento.txt")
        result = self.extractor._extract_from_txt(file_path)
        
        self.assertEqual(result, "Contenido del archivo TXT")
        mock_file.assert_called_once()
    
    @patch('builtins.open', side_effect=UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid'))
    def test_extract_from_txt_encoding_fallback(self, mock_file):
        """Test de fallback de encoding en archivos TXT."""
        # Simular que UTF-8 falla pero latin-1 funciona
        mock_file.side_effect = [
            UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid'),
            mock_open(read_data="Contenido con encoding alternativo").return_value
        ]
        
        file_path = Path("documento.txt")
        result = self.extractor._extract_from_txt(file_path)
        
        # Debería intentar múltiples encodings
        self.assertTrue(mock_file.call_count >= 2)
    
    @patch('organizador_documentos.core.content_extractor.pdfplumber')
    def test_extract_from_pdf(self, mock_pdfplumber):
        """Test de extracción de archivo PDF."""
        # Mock del PDF
        mock_page = Mock()
        mock_page.extract_text.return_value = "Texto de la página"
        mock_page.extract_tables.return_value = [
            [["Celda1", "Celda2"], ["Celda3", "Celda4"]]
        ]
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        file_path = Path("documento.pdf")
        result = self.extractor._extract_from_pdf(file_path)
        
        self.assertIn("Texto de la página", result)
        self.assertIn("Celda1 Celda2", result)
        mock_pdfplumber.open.assert_called_once_with(file_path)
    
    @patch('organizador_documentos.core.content_extractor.Document')
    def test_extract_from_docx(self, mock_document_class):
        """Test de extracción de archivo DOCX."""
        # Mock del documento
        mock_paragraph = Mock()
        mock_paragraph.text = "Párrafo de prueba"
        
        mock_cell = Mock()
        mock_cell.text = "Celda de tabla"
        mock_row = Mock()
        mock_row.cells = [mock_cell]
        mock_table = Mock()
        mock_table.rows = [mock_row]
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph]
        mock_doc.tables = [mock_table]
        
        mock_document_class.return_value = mock_doc
        
        file_path = Path("documento.docx")
        result = self.extractor._extract_from_docx(file_path)
        
        self.assertIn("Párrafo de prueba", result)
        self.assertIn("Celda de tabla", result)
        mock_document_class.assert_called_once_with(file_path)
    
    @patch('organizador_documentos.core.content_extractor.load_workbook')
    def test_extract_from_xlsx(self, mock_load_workbook):
        """Test de extracción de archivo XLSX."""
        # Mock del workbook
        mock_sheet = Mock()
        mock_sheet.iter_rows.return_value = [
            ("Valor1", "Valor2", None),
            (123, "Texto", 45.67)
        ]
        mock_sheet.max_row = 2
        
        mock_workbook = Mock()
        mock_workbook.sheetnames = ["Hoja1"]
        mock_workbook.__getitem__.return_value = mock_sheet
        
        mock_load_workbook.return_value = mock_workbook
        
        file_path = Path("documento.xlsx")
        result = self.extractor._extract_from_xlsx(file_path)
        
        self.assertIn("Valor1", result)
        self.assertIn("123", result)
        self.assertIn("45.67", result)
        mock_load_workbook.assert_called_once()
    
    def test_extract_text_unsupported_extension(self):
        """Test de extracción con extensión no soportada."""
        file_path = Path("archivo.xyz")
        result = self.extractor.extract_text(file_path)
        
        self.assertEqual(result, "")
    
    @patch('organizador_documentos.core.content_extractor.logger')
    def test_extract_text_with_exception(self, mock_logger):
        """Test de manejo de excepciones durante extracción."""
        # Simular excepción en extractor TXT
        with patch('builtins.open', side_effect=Exception("Error de prueba")):
            file_path = Path("documento.txt")
            result = self.extractor.extract_text(file_path)
            
            self.assertEqual(result, "")
            mock_logger.error.assert_called_once()
    
    def test_get_supported_extensions(self):
        """Test de obtención de extensiones soportadas."""
        extensions = self.extractor.get_supported_extensions()
        
        expected_extensions = ['.pdf', '.docx', '.xlsx', '.txt']
        for ext in expected_extensions:
            self.assertIn(ext, extensions)
    
    def test_get_extraction_stats(self):
        """Test de obtención de estadísticas."""
        stats = self.extractor.get_extraction_stats()
        
        expected_keys = [
            "total_extractions", "successful_extractions", 
            "failed_extractions", "avg_extraction_time"
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], (int, float))


if __name__ == '__main__':
    unittest.main()