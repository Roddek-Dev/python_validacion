"""
Tests para el clasificador de archivos.
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from organizador_documentos.core.file_classifier import FileClassifier
from organizador_documentos.utils.constants import DEFAULT_CONFIG


class TestFileClassifier(unittest.TestCase):
    """Tests para FileClassifier."""
    
    def setUp(self):
        """Configuración inicial para tests."""
        self.config = DEFAULT_CONFIG.copy()
        self.config.update({
            "categories": {
                "01": "01 Requisición del Personal",
                "02": "02 Formato Hoja de Vida Unica",
                "06": "06 Afiliación ARL",
                "10": "10 Documento de Identidad",
            },
            "keywords": {
                "01": ["requisicion", "solicitud personal", "formato requisicion"],
                "02": ["hoja de vida", "cv", "curriculum"],
                "06": ["arl", "afiliacion arl", "riesgos laborales"],
                "10": ["documento de identidad", "cedula de ciudadania", "cc"],
            }
        })
        
        # Mocks para extractores
        self.mock_content_extractor = Mock()
        self.mock_ocr_processor = Mock()
        
        self.classifier = FileClassifier(
            self.config, 
            self.mock_content_extractor, 
            self.mock_ocr_processor
        )
    
    def test_normalize_text(self):
        """Test de normalización de texto."""
        test_cases = [
            ("HOJA DE VIDA", "hoja de vida"),
            ("Cédula-de_Ciudadanía", "cedula de ciudadania"),
            ("REQUISICIÓN.pdf", "requisicion pdf"),
            ("", ""),
            (None, ""),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.classifier._normalize_text(input_text)
                self.assertEqual(result, expected)
    
    def test_get_keyword_score_critical(self):
        """Test de puntuación para keywords críticas."""
        # Keyword crítica específica para categoría 06
        score, type_desc = self.classifier._get_keyword_score("afiliacion arl", "06")
        self.assertEqual(score, 10)  # Normal keyword del config
        
        # Keyword genérica en categoría crítica
        score, type_desc = self.classifier._get_keyword_score("certificado", "06")
        self.assertEqual(score, 2)  # Penalizada
        
        # Keyword normal en categoría normal
        score, type_desc = self.classifier._get_keyword_score("hoja de vida", "02")
        self.assertEqual(score, 10)  # Normal
    
    def test_score_by_filename(self):
        """Test de puntuación por nombre de archivo."""
        test_cases = [
            ("HOJA DE VIDA CARLOS.pdf", {"02": 10}),
            ("REQUISICION PERSONAL.docx", {"01": 10}),
            ("CEDULA CIUDADANIA.jpg", {"10": 10}),
            ("archivo_sin_keywords.txt", {}),
        ]
        
        for filename, expected_scores in test_cases:
            with self.subTest(filename=filename):
                normalized_filename = self.classifier._normalize_text(filename)
                scores, keywords = self.classifier._score_by_filename(normalized_filename)
                
                for cat_id, expected_score in expected_scores.items():
                    self.assertGreaterEqual(scores[cat_id], expected_score)
    
    def test_extract_user_from_folder(self):
        """Test de extracción de usuario de carpetas."""
        test_cases = [
            ("BAUTISTA GONZÁLEZ CARLOS ANDRÉS - CC 1023902294", "Bautista_Gonzalez_Carlos_Andres"),
            ("MARIA LOPEZ CC 12345678", "Maria_Lopez"),
            ("JUAN PEREZ - 87654321", "Juan_Perez"),
            ("carpeta_sin_patron", None),
        ]
        
        origen_path = Path("/origen")
        
        for folder_name, expected_user in test_cases:
            with self.subTest(folder_name=folder_name):
                file_path = origen_path / folder_name / "documento.pdf"
                user = self.classifier.extract_user_from_folder(file_path, origen_path)
                self.assertEqual(user, expected_user)
    
    def test_classify_file_by_filename(self):
        """Test de clasificación por nombre de archivo."""
        # Mock para que no use contenido ni OCR
        self.config["enable_content_search"] = False
        self.config["enable_ocr"] = False
        
        origen_path = Path("/origen")
        file_path = origen_path / "HOJA_DE_VIDA_CARLOS.pdf"
        
        category, reason, score, keywords = self.classifier.classify_file(file_path, origen_path)
        
        self.assertEqual(category, "02")
        self.assertEqual(reason, "nombre_archivo")
        self.assertGreater(score, 0)
        self.assertTrue(any("filename:" in kw for kw in keywords))
    
    def test_classify_file_pending(self):
        """Test de archivo que va a pendientes."""
        # Mock para que no use contenido ni OCR
        self.config["enable_content_search"] = False
        self.config["enable_ocr"] = False
        
        origen_path = Path("/origen")
        file_path = origen_path / "archivo_sin_keywords.pdf"
        
        category, reason, score, keywords = self.classifier.classify_file(file_path, origen_path)
        
        self.assertEqual(category, "Pendientes_Revisar")
        self.assertIn(reason, ["baja_confianza", "sin_coincidencias"])
    
    @patch('organizador_documentos.core.file_classifier.logger')
    def test_classify_file_with_content(self, mock_logger):
        """Test de clasificación con análisis de contenido."""
        self.config["enable_content_search"] = True
        self.config["extensions_for_content_search"] = [".pdf"]
        
        # Mock del extractor de contenido
        self.mock_content_extractor.can_extract.return_value = True
        self.mock_content_extractor.extract_text.return_value = "Este es un documento de requisición de personal"
        
        origen_path = Path("/origen")
        file_path = origen_path / "documento.pdf"
        
        category, reason, score, keywords = self.classifier.classify_file(file_path, origen_path)
        
        self.assertEqual(category, "01")
        self.assertEqual(reason, "contenido")
        self.assertTrue(any("content:" in kw for kw in keywords))
    
    def test_clean_user_name(self):
        """Test de limpieza de nombres de usuario."""
        test_cases = [
            ("CARLOS ANDRÉS BAUTISTA", "Carlos_Andres_Bautista"),
            ("maría lópez garcía", "Maria_Lopez_Garcia"),
            ("JUAN-CARLOS PÉREZ", "Juan_Carlos_Perez"),
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                result = self.classifier._clean_user_name(input_name)
                self.assertEqual(result, expected)
    
    def test_determine_reason(self):
        """Test de determinación de razón de clasificación."""
        test_cases = [
            (["filename:hoja de vida"], "nombre_archivo"),
            (["folder:requisicion"], "carpeta_padre"),
            (["content:cedula"], "contenido"),
            (["ocr:arl"], "ocr"),
            (["filename:cv", "folder:documentos"], "carpeta_padre"),  # Prioridad a carpeta
        ]
        
        for keywords, expected_reason in test_cases:
            with self.subTest(keywords=keywords):
                reason = self.classifier._determine_reason(keywords)
                self.assertEqual(reason, expected_reason)
    
    def test_stats_tracking(self):
        """Test de seguimiento de estadísticas."""
        initial_stats = self.classifier.get_stats()
        
        # Simular clasificación
        self.config["enable_content_search"] = False
        self.config["enable_ocr"] = False
        
        origen_path = Path("/origen")
        file_path = origen_path / "HOJA_DE_VIDA.pdf"
        
        self.classifier.classify_file(file_path, origen_path)
        
        updated_stats = self.classifier.get_stats()
        
        self.assertEqual(updated_stats["classifications"], initial_stats["classifications"] + 1)
        self.assertEqual(updated_stats["filename_matches"], initial_stats["filename_matches"] + 1)


if __name__ == '__main__':
    unittest.main()