"""
Clasificador de archivos usando algoritmo de puntuación jerárquico.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ..utils.constants import (
    GENERIC_KEYWORDS, CRITICAL_SPECIFIC_KEYWORDS, USER_PATTERNS
)

logger = logging.getLogger(__name__)


class FileClassifier:
    """Clasifica archivos usando algoritmo de puntuación basado en palabras clave."""
    
    def __init__(self, config: Dict, content_extractor, ocr_processor):
        self.config = config
        self.content_extractor = content_extractor
        self.ocr_processor = ocr_processor
        self.categories = config.get("categories", {})
        self.keywords = config.get("keywords", {})
        self.confidence_threshold = config.get("confidence_threshold", 15)
        
        # Estadísticas de clasificación
        self.stats = {
            "classifications": 0,
            "filename_matches": 0,
            "folder_matches": 0,
            "content_matches": 0,
            "ocr_matches": 0,
            "pending_files": 0,
            "ambiguous_files": 0
        }
    
    def classify_file(self, file_path: Path, origen_path: Path) -> Tuple[str, str, int, List[str]]:
        """
        Clasifica un archivo usando algoritmo de puntuación jerárquico.
        
        Args:
            file_path: Ruta al archivo
            origen_path: Ruta base de origen
            
        Returns:
            Tupla con (categoria_id, razon, puntuacion, keywords_encontradas)
        """
        self.stats["classifications"] += 1
        
        filename = file_path.name
        normalized_filename = self._normalize_text(filename)
        
        scores = {cat_id: 0 for cat_id in self.categories.keys()}
        found_keywords = []
        
        # 1. Clasificación por nombre de archivo
        filename_score, filename_keywords = self._score_by_filename(normalized_filename)
        for cat_id, score in filename_score.items():
            scores[cat_id] += score
        found_keywords.extend(filename_keywords)
        
        if filename_keywords:
            self.stats["filename_matches"] += 1
        
        # 2. Clasificación por carpeta padre
        folder_score, folder_keywords = self._score_by_folder(file_path)
        for cat_id, score in folder_score.items():
            scores[cat_id] += score
        found_keywords.extend(folder_keywords)
        
        if folder_keywords:
            self.stats["folder_matches"] += 1
        
        # 3. Análisis de contenido (opcional)
        if self._should_analyze_content(file_path):
            content_score, content_keywords = self._score_by_content(file_path)
            for cat_id, score in content_score.items():
                scores[cat_id] += score
            found_keywords.extend(content_keywords)
            
            if content_keywords:
                self.stats["content_matches"] += 1
        
        # 4. OCR (opcional)
        if self._should_use_ocr(file_path):
            ocr_score, ocr_keywords = self._score_by_ocr(file_path)
            for cat_id, score in ocr_score.items():
                scores[cat_id] += score
            found_keywords.extend(ocr_keywords)
            
            if ocr_keywords:
                self.stats["ocr_matches"] += 1
        
        # 5. Determinar clasificación final
        return self._determine_final_classification(scores, found_keywords, filename)
    
    def extract_user_from_folder(self, file_path: Path, origen_path: Path) -> Optional[str]:
        """
        Extrae nombre de usuario de la estructura de carpetas.
        
        Args:
            file_path: Ruta al archivo
            origen_path: Ruta base de origen
            
        Returns:
            Nombre de usuario limpio o None
        """
        try:
            relative_path = file_path.relative_to(origen_path)
        except ValueError:
            return None
        
        path_parts = relative_path.parts
        
        # Buscar en las primeras partes de la ruta
        for part in path_parts[:3]:
            for pattern in USER_PATTERNS:
                match = re.search(pattern, part, re.IGNORECASE)
                if match:
                    user_name = match.group(1).strip()
                    return self._clean_user_name(user_name)
        
        return None
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparación."""
        if not text:
            return ""
        
        text = text.lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r"[-._\s$$$$\[\]{}]+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def _clean_user_name(self, name: str) -> str:
        """Limpia y formatea el nombre de usuario."""
        clean_name = self._normalize_text(name)
        clean_name = clean_name.replace(" ", "_")
        clean_name = "_".join(word.capitalize() for word in clean_name.split("_"))
        return clean_name
    
    def _get_keyword_score(self, keyword: str, category_id: str) -> Tuple[int, str]:
        """Determina puntuación de keyword basada en especificidad."""
        normalized_keyword = self._normalize_text(keyword)
        
        # Verificar keywords críticas específicas
        if category_id in CRITICAL_SPECIFIC_KEYWORDS:
            for critical in CRITICAL_SPECIFIC_KEYWORDS[category_id]:
                if self._normalize_text(critical) in normalized_keyword or normalized_keyword in self._normalize_text(critical):
                    return 30, "crítica específica"
        
        # Penalizar keywords genéricas en categorías críticas
        if category_id in ["06", "10"]:
            for generic in GENERIC_KEYWORDS:
                if self._normalize_text(generic) in normalized_keyword:
                    return 2, "genérica penalizada"
        
        # Keywords genéricas normales
        for generic in GENERIC_KEYWORDS:
            if self._normalize_text(generic) in normalized_keyword:
                return 5, "genérica"
        
        # Keywords normales del config
        return 10, "normal"
    
    def _score_by_filename(self, normalized_filename: str) -> Tuple[Dict[str, int], List[str]]:
        """Puntúa basado en palabras clave en el nombre del archivo."""
        scores = {cat_id: 0 for cat_id in self.categories.keys()}
        found_keywords = []
        
        for cat_id, cat_keywords in self.keywords.items():
            for keyword in cat_keywords:
                normalized_keyword = self._normalize_text(keyword)
                if normalized_keyword in normalized_filename:
                    keyword_score, _ = self._get_keyword_score(keyword, cat_id)
                    scores[cat_id] += keyword_score
                    found_keywords.append(f"filename:{keyword}")
        
        return scores, found_keywords
    
    def _score_by_folder(self, file_path: Path) -> Tuple[Dict[str, int], List[str]]:
        """Puntúa basado en palabras clave en la carpeta padre."""
        scores = {cat_id: 0 for cat_id in self.categories.keys()}
        found_keywords = []
        
        parent_folder = file_path.parent.name
        normalized_parent = self._normalize_text(parent_folder)
        
        for cat_id, cat_keywords in self.keywords.items():
            for keyword in cat_keywords:
                normalized_keyword = self._normalize_text(keyword)
                if normalized_keyword in normalized_parent:
                    keyword_score, _ = self._get_keyword_score(keyword, cat_id)
                    folder_score = int(keyword_score * 0.7)  # Reducir peso de carpeta
                    scores[cat_id] += folder_score
                    found_keywords.append(f"folder:{keyword}")
        
        return scores, found_keywords
    
    def _score_by_content(self, file_path: Path) -> Tuple[Dict[str, int], List[str]]:
        """Puntúa basado en análisis de contenido."""
        scores = {cat_id: 0 for cat_id in self.categories.keys()}
        found_keywords = []
        
        content_text = self.content_extractor.extract_text(file_path)
        if not content_text:
            return scores, found_keywords
        
        normalized_content = self._normalize_text(content_text)
        
        for cat_id, cat_keywords in self.keywords.items():
            for keyword in cat_keywords:
                normalized_keyword = self._normalize_text(keyword)
                if normalized_keyword in normalized_content:
                    keyword_score, _ = self._get_keyword_score(keyword, cat_id)
                    content_score = max(2, int(keyword_score * 0.5))  # Reducir peso de contenido
                    scores[cat_id] += content_score
                    found_keywords.append(f"content:{keyword}")
        
        return scores, found_keywords
    
    def _score_by_ocr(self, file_path: Path) -> Tuple[Dict[str, int], List[str]]:
        """Puntúa basado en OCR."""
        scores = {cat_id: 0 for cat_id in self.categories.keys()}
        found_keywords = []
        
        ocr_text = self.ocr_processor.extract_text_with_ocr(file_path)
        if not ocr_text.strip():
            return scores, found_keywords
        
        normalized_ocr = self._normalize_text(ocr_text)
        
        for cat_id, cat_keywords in self.keywords.items():
            for keyword in cat_keywords:
                normalized_keyword = self._normalize_text(keyword)
                if normalized_keyword in normalized_ocr:
                    keyword_score, _ = self._get_keyword_score(keyword, cat_id)
                    ocr_score = max(2, int(keyword_score * 0.8))  # Peso medio para OCR
                    scores[cat_id] += ocr_score
                    found_keywords.append(f"ocr:{keyword}")
        
        return scores, found_keywords
    
    def _should_analyze_content(self, file_path: Path) -> bool:
        """Determina si se debe analizar el contenido del archivo."""
        if not self.config.get("enable_content_search", False):
            return False
        
        extension = file_path.suffix.lower()
        supported_extensions = self.config.get("extensions_for_content_search", [])
        
        return (extension in supported_extensions and 
                self.content_extractor.can_extract(file_path))
    
    def _should_use_ocr(self, file_path: Path) -> bool:
        """Determina si se debe usar OCR en el archivo."""
        if not self.config.get("enable_ocr", False):
            return False
        
        return self.ocr_processor.can_process_ocr()
    
    def _determine_final_classification(self, scores: Dict[str, int], 
                                      found_keywords: List[str], 
                                      filename: str) -> Tuple[str, str, int, List[str]]:
        """Determina la clasificación final basada en puntuaciones."""
        max_score = max(scores.values()) if scores.values() else 0
        
        if max_score >= self.confidence_threshold:
            top_categories = [cat_id for cat_id, score in scores.items() if score == max_score]
            
            # Verificar categorías competidoras
            competing_categories = [
                cat_id for cat_id, score in scores.items()
                if score >= max_score * 0.7 and score > 15
            ]
            
            # Aplicar penalización por ambigüedad
            if len(competing_categories) > 1:
                penalty = len(competing_categories) * 3
                logger.warning(f"Penalización por ambigüedad: -{penalty} puntos para {filename}")
                max_score -= penalty
                self.stats["ambiguous_files"] += 1
            
            # Clasificación exitosa
            if max_score >= self.confidence_threshold and len(top_categories) == 1:
                category = top_categories[0]
                reason = self._determine_reason(found_keywords)
                return category, reason, max_score, found_keywords
            
            # Empate después de penalizaciones
            elif len(top_categories) > 1:
                logger.warning(f"Empate después de penalizaciones en {top_categories} para: {filename}")
                self.stats["pending_files"] += 1
                return "Pendientes_Revisar", "empate_post_penalizacion", max_score, found_keywords
        
        # Archivo sin clasificar
        reason = "baja_confianza" if max_score > 0 else "sin_coincidencias"
        logger.warning(f"Archivo sin clasificar ({max_score} < {self.confidence_threshold}): {filename}")
        self.stats["pending_files"] += 1
        return "Pendientes_Revisar", reason, max_score, found_keywords
    
    def _determine_reason(self, found_keywords: List[str]) -> str:
        """Determina la razón de clasificación basada en keywords encontradas."""
        if any(kw.startswith("folder:") for kw in found_keywords):
            return "carpeta_padre"
        elif any(kw.startswith("content:") for kw in found_keywords):
            return "contenido"
        elif any(kw.startswith("ocr:") for kw in found_keywords):
            return "ocr"
        else:
            return "nombre_archivo"
    
    def get_stats(self) -> Dict[str, int]:
        """Retorna estadísticas de clasificación."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reinicia las estadísticas."""
        for key in self.stats:
            self.stats[key] = 0