#!/usr/bin/env python3
"""
Organizador de Documentos - Script de automatización para clasificación de documentos

Clasifica, organiza y renombra documentos usando algoritmo de puntuación basado en:
- Palabras clave en nombres de archivo y carpetas
- Análisis de contenido (opcional)
- OCR para imágenes y PDFs escaneados (opcional)

Uso:
    python organizador_documentos.py --origen /ruta/origen --destino /ruta/destino
    python organizador_documentos.py --enable-ocr --enable-content --verbose
"""

import argparse
import csv
import hashlib
import logging
import os
import re
import shutil
import sys
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import yaml
from tqdm import tqdm

# Importaciones opcionales
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from openpyxl import load_workbook
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


class DocumentOrganizer:
    """Clase principal para la organización de documentos."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.processed_hashes: Set[str] = set()
        self.results: List[Dict] = []
        self.lock = threading.Lock()
        self.logger = self._setup_logger()
        self.stats = {
            "total_files": 0, "classified": 0, "duplicates": 0,
            "pending": 0, "errors": 0, "ocr_used": 0, "content_analysis_used": 0,
            "users_created": 0
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Configura el sistema de logging."""
        logger = logging.getLogger("DocumentOrganizer")
        logger.setLevel(logging.DEBUG if self.config.get("verbose", False) else logging.INFO)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_file = self.config.get("log_file", "proceso.log")
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
        except ImportError:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")

        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(threadName)s] - %(funcName)s:%(lineno)d - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger
    
    def normalize_text(self, text: str) -> str:
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
    
    def extract_user_from_folder(self, file_path: Path, origen_path: Path) -> Optional[str]:
        """Extrae nombre de usuario de la estructura de carpetas."""
        # Obtener la ruta relativa desde la carpeta origen
        try:
            relative_path = file_path.relative_to(origen_path)
        except ValueError:
            return None
        
        # Dividir la ruta en partes
        path_parts = relative_path.parts
        
        # Patrones para detectar carpetas de usuario
        user_patterns = [
            r"([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑ\s]+)\s+-\s+CC\s+\d+",  # "NOMBRE APELLIDO - CC 123456789"
            r"([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑ\s]+)\s+CC\s+\d+",      # "NOMBRE APELLIDO CC 123456789"
            r"([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑ\s]+)\s+-\s+\d+",       # "NOMBRE APELLIDO - 123456789"
        ]
        
        # Buscar en las primeras partes de la ruta (carpetas más cercanas al origen)
        for part in path_parts[:3]:  # Solo las primeras 3 carpetas
            for pattern in user_patterns:
                match = re.search(pattern, part, re.IGNORECASE)
                if match:
                    user_name = match.group(1).strip()
                    return self._clean_user_name(user_name)
        
        return None
    
    def _clean_user_name(self, name: str) -> str:
        """Limpia y formatea el nombre de usuario."""
        # Normalizar y limpiar
        clean_name = self.normalize_text(name)
        # Reemplazar espacios con guiones bajos
        clean_name = clean_name.replace(" ", "_")
        # Capitalizar primera letra de cada palabra
        clean_name = "_".join(word.capitalize() for word in clean_name.split("_"))
        return clean_name

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calcula hash SHA256 del archivo."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculando hash para {file_path}: {e}")
            return ""
    
    def is_pdf_scanned(self, file_path: Path) -> bool:
        """Detecta si un PDF es escaneado y necesita OCR."""
        if not PDF_AVAILABLE:
            return True

        try:
            text_content = ""
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages[:2]):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text

            text_len = len(text_content.strip())
            if text_len < self.config.get("scanned_pdf_threshold", 100):
                return True

            weird_chars = sum(1 for c in text_content if ord(c) > 127 or c in "□■▪▫")
            weird_ratio = weird_chars / max(text_len, 1)
            if weird_ratio > 0.3:
                return True

            space_ratio = text_content.count(" ") / max(text_len, 1)
            if space_ratio > 0.7:
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Error verificando PDF escaneado {file_path}: {e}")
            return True

    def preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Pre-procesa imagen para mejorar OCR."""
        try:
            if image.mode != "L":
                image = image.convert("L")

            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            image = image.filter(ImageFilter.SHARPEN)

            width, height = image.size
            if width < 1200 or height < 1200:
                scale_factor = max(1200 / width, 1200 / height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            return image
        except Exception as e:
            self.logger.error(f"Error pre-procesando imagen: {e}")
            return image

    def extract_text_from_pdf(self, file_path: Path, max_pages: int = 1) -> str:
        """Extrae texto de PDF."""
        if not PDF_AVAILABLE:
            return ""
        
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages[:max_pages]):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

                        tables = page.extract_tables()
                        for table in tables:
                            for row in table:
                                if row:
                                    text += " ".join(str(cell) for cell in row if cell) + "\n"
                    except Exception as page_error:
                        self.logger.warning(f"Error procesando página {i+1} de {file_path}: {page_error}")
                        continue
            return text
        except Exception as e:
            self.logger.error(f"Error extrayendo texto de PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: Path) -> str:
        """Extrae texto de DOCX."""
        if not DOCX_AVAILABLE:
            return ""
        
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extrayendo texto de DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_xlsx(self, file_path: Path, max_sheets: int = 1) -> str:
        """Extrae texto de XLSX."""
        if not XLSX_AVAILABLE:
            return ""
        
        try:
            workbook = load_workbook(file_path, data_only=True)
            text = ""
            for i, sheet_name in enumerate(workbook.sheetnames[:max_sheets]):
                sheet = workbook[sheet_name]
                for row in sheet.iter_rows(values_only=True):
                    for cell in row:
                        if cell is not None:
                            text += str(cell) + " "
                text += "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extrayendo texto de XLSX {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: Path) -> str:
        """Extrae texto de TXT."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error extrayendo texto de TXT {file_path}: {e}")
            return ""
    
    def extract_text_with_ocr(self, file_path: Path) -> str:
        """OCR mejorado con pre-procesamiento."""
        if not OCR_AVAILABLE:
            self.logger.warning("OCR no disponible - instale Pillow y pytesseract")
            return ""

        try:
            extension = file_path.suffix.lower()

            if extension == ".pdf":
                if not self.is_pdf_scanned(file_path):
                    return self.extract_text_from_pdf(file_path)

                if not PDF2IMAGE_AVAILABLE:
                    self.logger.warning("pdf2image no disponible para OCR en PDFs")
                    return ""
                
                try:
                    pages = convert_from_path(str(file_path), dpi=300, first_page=1, last_page=1)
                    text_parts = []
                    ocr_config = "--oem 3 --psm 6 -l " + self.config.get("ocr_languages", "spa")

                    for page_img in pages:
                        processed_img = self.preprocess_image_for_ocr(page_img)
                        page_text = pytesseract.image_to_string(processed_img, config=ocr_config)
                        if page_text.strip():
                            text_parts.append(page_text)

                    result = "\n".join(text_parts)
                    if result.strip():
                        self.stats["ocr_used"] += 1
                        self.logger.debug(f"OCR exitoso en PDF: {file_path.name} ({len(result)} chars)")
                    return result

                except Exception as conv_error:
                    self.logger.error(f"Error convirtiendo PDF para OCR {file_path}: {conv_error}")
                    return ""
            
            elif extension in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]:
                image = Image.open(file_path)
                processed_image = self.preprocess_image_for_ocr(image)
                ocr_config = "--oem 3 --psm 6 -l " + self.config.get("ocr_languages", "spa")
                text = pytesseract.image_to_string(processed_image, config=ocr_config)

                if text.strip():
                    self.stats["ocr_used"] += 1
                    self.logger.debug(f"OCR exitoso en imagen: {file_path.name} ({len(text)} chars)")
                return text
            
        except Exception as e:
            self.logger.error(f"Error en OCR mejorado para {file_path}: {e}")
        
        return ""
    
    def extract_content_text(self, file_path: Path) -> str:
        """Extrae texto del contenido según extensión."""
        extension = file_path.suffix.lower()
        
        if extension == ".pdf":
            return self.extract_text_from_pdf(file_path, self.config.get("pdf_pages_to_read", 1))
        elif extension == ".docx":
            return self.extract_text_from_docx(file_path)
        elif extension == ".xlsx":
            return self.extract_text_from_xlsx(file_path)
        elif extension == ".txt":
            return self.extract_text_from_txt(file_path)
        return ""
    
    def get_keyword_score(self, keyword: str, category_id: str) -> Tuple[int, str]:
        """Determina puntuación de keyword basada en especificidad."""
        specific_keywords = {
            "00": ["check list", "lista chequeo", "formato check list"],
            "01": ["requisicion personal", "solicitud personal", "requisicion"],
            "02": ["hoja vida", "curriculum vitae", "cv profesional"],
            "03": ["diploma", "acta grado", "titulo profesional", "certificado estudios"],
            "04": ["certificacion laboral", "experiencia laboral", "constancia laboral"],
            "05": ["afiliacion eps", "salud eps"],
            "06": ["afiliacion arl", "riesgos laborales"],
            "07": ["caja compensacion", "ccf"],
            "08": ["fondo pension", "afiliacion pension"],
            "09": ["contrato trabajo", "contrato laboral"],
            "10": ["cedula ciudadania", "documento identidad"],
            "11": ["examen medico ingreso", "aptitud medica"],
            "12": ["pgn", "procuraduria", "antecedentes disciplinarios"],
            "13": ["cgr", "contraloria", "antecedentes fiscales"],
            "14": ["ponal", "antecedentes judiciales"],
            "15": ["medidas correctivas ponal"],
            "16": ["certificacion bancaria", "cuenta bancaria"],
            "17": ["licencia conduccion"],
            "18": ["tarjeta profesional", "matricula profesional"],
            "19": ["induccion corporativa", "constancia induccion"],
            "20": ["foto"],
            "21": ["tratamiento datos personales", "habeas data", "autorizacion datos"],
            "22": ["inhabilidad incompatibilidad", "declaracion juramentada"],
            "23": ["otros documentos ingreso"],
            "24": ["examen medico periodico", "reubicacion medica"],
            "25": ["permisos", "licencia no remunerada"],
            "27": ["incapacidad medica"],
            "28": ["vacaciones", "solicitud vacaciones"],
            "29": ["encargo"],
            "30": ["cesantias"],
            "31": ["certificaciones laborales"],
            "32": ["deduccion retefuente", "retencion fuente"],
            "33": ["ingresos retenciones"],
            "34": ["verificacion titulo", "poligrafo"],
            "35": ["reinducciones"],
            "36": ["carnet vacunas", "vacunacion"],
            "37": ["paz salvo", "acta retiro", "examen egreso"],
            "38": ["acta entrega puesto"],
            "39": ["validacion"],
            "40": ["proceso disciplinario", "memorando disciplinario"],
            "41": ["otros documentos"],
            "42": ["evaluacion desempeno", "evaluacion rendimiento"],
        }

        generic_keywords = [
            "carlos", "bautista", "andres", "gonzalez", "certificado",
            "certificacion", "formato", "documento", "ingreso", "laboral", "personal"
        ]

        normalized_keyword = self.normalize_text(keyword)

        if category_id in specific_keywords:
            for specific in specific_keywords[category_id]:
                if self.normalize_text(specific) == normalized_keyword:
                    return 20, "específica"

        for generic in generic_keywords:
            if self.normalize_text(generic) in normalized_keyword:
                return 5, "genérica"

        return 10, "normal"

    def classify_file(self, file_path: Path) -> Tuple[str, str, int, List[str]]:
        """Clasificación con sistema de puntuación jerárquico."""
        filename = file_path.name
        normalized_filename = self.normalize_text(filename)
        categories = self.config["categories"]
        keywords = self.config["keywords"]
        
        scores = {cat_id: 0 for cat_id in categories.keys()}
        found_keywords = []
        
        # Clasificación por palabras clave en nombre de archivo
        for cat_id, cat_keywords in keywords.items():
            filename_score = 0
            for keyword in cat_keywords:
                normalized_keyword = self.normalize_text(keyword)
                if normalized_keyword in normalized_filename:
                    keyword_score, keyword_type = self.get_keyword_score(keyword, cat_id)
                    filename_score += keyword_score
                    found_keywords.append(f"filename:{keyword}")

            scores[cat_id] += filename_score

        # Clasificación por carpeta padre
        parent_folder = file_path.parent.name
        normalized_parent = self.normalize_text(parent_folder)
        
        for cat_id, cat_keywords in keywords.items():
            folder_score = 0
            for keyword in cat_keywords:
                normalized_keyword = self.normalize_text(keyword)
                if normalized_keyword in normalized_parent:
                    keyword_score, keyword_type = self.get_keyword_score(keyword, cat_id)
                    folder_score += int(keyword_score * 0.7)
                    found_keywords.append(f"folder:{keyword}")

            scores[cat_id] += folder_score

        # Análisis de contenido
        if (self.config.get("enable_content_search", False) and 
            file_path.suffix.lower() in self.config.get("extensions_for_content_search", [])):
            
            content_text = self.extract_content_text(file_path)
            if content_text:
                self.stats["content_analysis_used"] += 1
                normalized_content = self.normalize_text(content_text)
                
                for cat_id, cat_keywords in keywords.items():
                    content_score = 0
                    for keyword in cat_keywords:
                        normalized_keyword = self.normalize_text(keyword)
                        if normalized_keyword in normalized_content:
                            keyword_score, keyword_type = self.get_keyword_score(keyword, cat_id)
                            content_score += max(2, int(keyword_score * 0.5))
                            found_keywords.append(f"content:{keyword}")

                    scores[cat_id] += content_score

        # OCR
        if self.config.get("enable_ocr", False):
            ocr_text = self.extract_text_with_ocr(file_path)

            if ocr_text.strip():
                normalized_ocr = self.normalize_text(ocr_text)
                
                for cat_id, cat_keywords in keywords.items():
                    ocr_score = 0
                    for keyword in cat_keywords:
                        normalized_keyword = self.normalize_text(keyword)
                        if normalized_keyword in normalized_ocr:
                            keyword_score, keyword_type = self.get_keyword_score(keyword, cat_id)
                            ocr_score += max(2, int(keyword_score * 0.8))
                            found_keywords.append(f"ocr:{keyword}")

                    scores[cat_id] += ocr_score

        max_score = max(scores.values()) if scores.values() else 0
        confidence_threshold = self.config.get("confidence_threshold", 15)

        if max_score >= confidence_threshold:
            top_categories = [cat_id for cat_id, score in scores.items() if score == max_score]
            
            competing_categories = [
                cat_id for cat_id, score in scores.items()
                if score >= max_score * 0.7 and score > 15
            ]

            if len(competing_categories) > 1:
                penalty = len(competing_categories) * 3
                self.logger.warning(f"Penalización por ambigüedad: -{penalty} puntos para {filename}")
                max_score -= penalty

            if max_score >= confidence_threshold and len(top_categories) == 1:
                category = top_categories[0]
                reason = self._determine_reason(scores[category], found_keywords)
                return category, reason, max_score, found_keywords
            elif len(top_categories) > 1:
                self.logger.warning(f"Empate después de penalizaciones en {top_categories} para: {filename}")
                return "Pendientes_Revisar", "empate_post_penalizacion", max_score, found_keywords

        reason = "baja_confianza" if max_score > 0 else "sin_coincidencias"
        self.logger.warning(f"Archivo sin clasificar ({max_score} < {confidence_threshold}): {filename}")
        return "Pendientes_Revisar", reason, max_score, found_keywords

    def _determine_reason(self, score: int, found_keywords: List[str]) -> str:
        """Determina la razón de clasificación."""
        if any(kw.startswith("folder:") for kw in found_keywords):
            return "carpeta_padre"
        elif any(kw.startswith("content:") for kw in found_keywords):
            return "contenido"
        elif any(kw.startswith("ocr:") for kw in found_keywords):
            return "ocr"
        else:
            return "nombre_archivo"
    
    def clean_filename(self, filename: str) -> str:
        """Limpia nombre de archivo eliminando caracteres inválidos."""
        invalid_chars = '<>:"/\\|?*'
        clean_name = filename
        for char in invalid_chars:
            clean_name = clean_name.replace(char, "_")
        
        max_length = self.config.get("max_filename_length", 100)
        if len(clean_name) > max_length:
            name_part, ext = os.path.splitext(clean_name)
            clean_name = name_part[: max_length - len(ext)] + ext
        
        return clean_name
    
    def generate_destination_path(self, original_path: Path, category_id: str, destino_path: Path, user_name: Optional[str] = None) -> Tuple[str, Path]:
        """Genera ruta de destino manteniendo el nombre original del archivo."""
        # Crear estructura: destino_path / usuario / categoria
        if (self.config.get("enable_user_organization", False) and user_name):
            # Crear carpeta de usuario en la raíz del destino
            user_folder = destino_path / user_name
            user_folder.mkdir(parents=True, exist_ok=True)
            
            # Crear carpeta de categoría dentro del usuario
            category_name = self.config["categories"][category_id]
            final_dest_folder = user_folder / category_name
            final_dest_folder.mkdir(parents=True, exist_ok=True)
            self.stats["users_created"] += 1
        else:
            # Estructura tradicional: destino_path / categoria
            category_name = self.config["categories"][category_id]
            final_dest_folder = destino_path / category_name
            final_dest_folder.mkdir(parents=True, exist_ok=True)

        # Mantener nombre original del archivo
        original_filename = original_path.name
        final_path = final_dest_folder / original_filename
        
        # Verificar si ya existe y añadir sufijo numérico si es necesario
        counter = 1
        while final_path.exists():
            name_part, ext = os.path.splitext(original_filename)
            new_filename = f"{name_part} ({counter}){ext}"
            final_path = final_dest_folder / new_filename
            counter += 1
        
        return final_path.name, final_dest_folder
    
    def process_file(self, file_path: Path, origen_path: Path, destino_path: Path) -> Dict:
        """Procesa un archivo individual."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "ruta_original": str(file_path.relative_to(origen_path)),
            "hash_sha256": "",
            "categoria_asignada": "",
            "razon_decision": "",
            "puntuacion": 0,
            "palabras_clave_encontradas": "",
            "ruta_destino": "",
            "estado": "",
            "mensaje_error": "",
        }
        
        try:
            file_hash = self.calculate_file_hash(file_path)
            result["hash_sha256"] = file_hash
            
            with self.lock:
                if file_hash in self.processed_hashes:
                    result["estado"] = "Duplicado"
                    result["categoria_asignada"] = "Duplicados"
                    self.stats["duplicates"] += 1
                    
                    duplicados_folder = destino_path / "Duplicados"
                    duplicados_folder.mkdir(exist_ok=True)
                    new_filename = file_path.name
                    dest_path = duplicados_folder / new_filename
                    
                    counter = 1
                    while dest_path.exists():
                        name_part, ext = os.path.splitext(new_filename)
                        new_filename = f"{name_part}_dup_{counter}{ext}"
                        dest_path = duplicados_folder / new_filename
                        counter += 1
                    
                    result["ruta_destino"] = str(dest_path.relative_to(destino_path))
                    
                    if not self.config.get("dry_run", True):
                        if self.config.get("default_mode", "copy") == "move":
                            shutil.move(str(file_path), str(dest_path))
                        else:
                            shutil.copy2(str(file_path), str(dest_path))
                    
                    self.logger.info(f"Archivo duplicado: {file_path.name}")
                    return result
                
                self.processed_hashes.add(file_hash)
            
            category_id, reason, score, keywords = self.classify_file(file_path)
            
            result["categoria_asignada"] = category_id
            result["razon_decision"] = reason
            result["puntuacion"] = score
            result["palabras_clave_encontradas"] = "; ".join(keywords)
            
            # Detectar usuario de la estructura de carpetas
            user_name = self.extract_user_from_folder(file_path, origen_path)
            
            if category_id == "Pendientes_Revisar":
                if (self.config.get("enable_user_organization", False) and user_name):
                    # Crear carpeta de usuario en la raíz del destino
                    user_folder = destino_path / user_name
                    user_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Crear carpeta Pendientes_Revisar dentro del usuario
                    dest_folder = user_folder / "Pendientes_Revisar"
                    dest_folder.mkdir(parents=True, exist_ok=True)
                    self.stats["users_created"] += 1
                else:
                    dest_folder = destino_path / "Pendientes_Revisar"
                    dest_folder.mkdir(parents=True, exist_ok=True)
                
                new_filename = file_path.name
                dest_path = dest_folder / new_filename
                self.stats["pending"] += 1
                
                counter = 1
                while dest_path.exists():
                    name_part, ext = os.path.splitext(file_path.name)
                    new_filename = f"{name_part}_pendiente_{counter}{ext}"
                    dest_path = dest_folder / new_filename
                    counter += 1
            else:
                new_filename, final_dest_folder = self.generate_destination_path(file_path, category_id, destino_path, user_name)
                dest_path = final_dest_folder / new_filename
                self.stats["classified"] += 1
            
            result["ruta_destino"] = str(dest_path.relative_to(destino_path))
            result["estado"] = "Clasificado"
            
            if not self.config.get("dry_run", True):
                if self.config.get("default_mode", "copy") == "move":
                    shutil.move(str(file_path), str(dest_path))
                else:
                    shutil.copy2(str(file_path), str(dest_path))
            
            if category_id == "Pendientes_Revisar":
                self.logger.warning(f"Archivo pendiente de revisar: {file_path.name} -> {reason}")
            else:
                self.logger.info(f"Archivo procesado: {file_path.name} -> {category_id} ({reason})")
            
        except Exception as e:
            result["estado"] = "Error"
            result["mensaje_error"] = str(e)
            self.logger.error(f"Error procesando {file_path}: {e}")
            self.stats["errors"] += 1
        
        return result
    
    def find_files(self, origen_path: Path) -> List[Path]:
        """Encuentra todos los archivos en carpeta origen y subcarpetas."""
        files = []
        supported_extensions = {".pdf", ".docx", ".xlsx", ".jpg", ".jpeg", ".png", ".txt", ".tiff", ".bmp"}
        
        for file_path in origen_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
        
        return files
    
    def organize_documents(self, origen_path: Path, destino_path: Path):
        """Organiza todos los documentos de la carpeta origen."""
        self.logger.info(f"Iniciando organización de documentos")
        self.logger.info(f"Origen: {origen_path}")
        self.logger.info(f"Destino: {destino_path}")
        self.logger.info(f"Modo: {'DRY RUN' if self.config.get('dry_run', True) else self.config.get('default_mode', 'copy').upper()}")
        
        destino_path.mkdir(parents=True, exist_ok=True)
        
        files = self.find_files(origen_path)
        self.stats["total_files"] = len(files)
        self.logger.info(f"Encontrados {len(files)} archivos para procesar")
        
        if not files:
            self.logger.warning("No se encontraron archivos para procesar")
            return
        
        num_threads = self.config.get("num_threads", 4)
        self.logger.info(f"Usando {num_threads} hilos para procesamiento")
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            with tqdm(total=len(files), desc="Procesando archivos") as pbar:
                future_to_file = {
                    executor.submit(self.process_file, file_path, origen_path, destino_path): file_path
                    for file_path in files
                }
                
                for future in as_completed(future_to_file):
                    result = future.result()
                    with self.lock:
                        self.results.append(result)
                    pbar.update(1)
        
        self.save_results_csv()
        self.print_summary()

    
    def save_results_csv(self):
        """Guarda resultados en archivo CSV."""
        csv_file = self.config.get("csv_file", "resultados.csv")
        
        try:
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                if self.results:
                    writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                    writer.writeheader()
                    writer.writerows(self.results)
            
            self.logger.info(f"Resultados guardados en: {csv_file}")
        except Exception as e:
            self.logger.error(f"Error guardando CSV: {e}")
    
    def print_summary(self):
        """Imprime resumen del procesamiento."""
        if not self.results:
            return
        
        total_files = len(self.results)
        
        estados = {}
        for result in self.results:
            estado = result["estado"]
            estados[estado] = estados.get(estado, 0) + 1
        
        categorias = {}
        pendientes_por_razon = {}
        for result in self.results:
            if result["estado"] == "Clasificado":
                categoria = result["categoria_asignada"]
                categorias[categoria] = categorias.get(categoria, 0) + 1
                
                if categoria == "Pendientes_Revisar":
                    razon = result["razon_decision"]
                    pendientes_por_razon[razon] = pendientes_por_razon.get(razon, 0) + 1
        
        print("\n" + "=" * 60)
        print("RESUMEN DE PROCESAMIENTO")
        print("=" * 60)
        print(f"Total de archivos procesados: {total_files}")
        print(f"Modo de ejecución: {'DRY RUN (simulación)' if self.config.get('dry_run', True) else 'REAL'}")
        
        print("\nPor estado:")
        for estado, count in estados.items():
            print(f"  {estado}: {count}")
        
        if categorias:
            print("\nPor categoría:")
            for categoria, count in sorted(categorias.items()):
                if categoria == "Pendientes_Revisar":
                    print(f"  {categoria}: {count}")
                else:
                    category_name = self.config["categories"].get(categoria, categoria)
                    print(f"  {category_name}: {count}")
        
        if pendientes_por_razon:
            print("\nArchivos pendientes de revisar por razón:")
            for razon, count in pendientes_por_razon.items():
                razon_desc = {
                    "baja_confianza": "Baja confianza (puntuación insuficiente)",
                    "sin_coincidencias": "Sin coincidencias (no se encontraron keywords)",
                    "empate_post_penalizacion": "Empate después de penalizaciones",
                    "sin_clasificar": "Sin clasificar (puntuación baja)",
                    "ambiguo": "Ambiguo (empate en categorías)",
                }.get(razon, razon)
                print(f"  {razon_desc}: {count}")
        
        duplicados = estados.get("Duplicado", 0)
        pendientes = categorias.get("Pendientes_Revisar", 0)
        errores = estados.get("Error", 0)
        
        print(f"\nArchivos duplicados encontrados: {duplicados}")
        print(f"Archivos pendientes de revisar: {pendientes}")
        print(f"Archivos con errores: {errores}")

        print(f"\nEstadísticas de procesamiento:")
        print(f"  Análisis de contenido usado: {self.stats['content_analysis_used']} archivos")
        print(f"  OCR usado: {self.stats['ocr_used']} archivos")
        print(f"  Carpetas de usuario creadas: {self.stats['users_created']}")
        print("=" * 60)


def load_config(config_path: Optional[str] = None) -> Dict:
    """Carga configuración desde archivo YAML o usa valores por defecto."""
    default_config = {
        "dry_run": True,
        "default_mode": "copy",
        "num_threads": 4,
        "confidence_threshold": 15,
        "enable_content_search": False,
        "pdf_pages_to_read": 1,
        "extensions_for_content_search": [".pdf", ".docx", ".txt", ".xlsx"],
        "enable_ocr": False,
        "ocr_languages": "spa",
        "ocr_max_pages": 1,
        "scanned_pdf_threshold": 50,
        "ocr_min_text_threshold": 30,
        "max_filename_length": 100,
        "single_file_name_categories": {"00": True, "02": True, "10": True, "20": True},
        "enable_user_organization": False,
    }

    try:
        candidate_path: Optional[Path] = None
        if config_path:
            candidate_path = Path(config_path)
        else:
            local_cfg = Path("config.yml")
            if local_cfg.exists():
                candidate_path = local_cfg
        if candidate_path and candidate_path.exists():
            with open(candidate_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
                default_config.update(user_config)
    except Exception as e:
        print(f"Error cargando configuración: {e}")
        print("Usando configuración por defecto")
    
    return default_config


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Organizador de Documentos - Automatización de clasificación de documentos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python organizador_documentos.py --origen /ruta/origen --destino /ruta/destino
  python organizador_documentos.py --enable-ocr --enable-content --verbose
        """,
    )

    default_origen = str(Path.home() / "Escritorio" / "Documentos_Desordenados")
    default_destino = str(Path.home() / "Escritorio" / "Documentos_Organizados")

    parser.add_argument("--origen", required=False, default=default_origen, help="Ruta a la carpeta de origen")
    parser.add_argument("--destino", required=False, default=default_destino, help="Ruta a la carpeta de destino")
    parser.add_argument("--config", help="Ruta al archivo config.yml")
    parser.add_argument("--mode", choices=["copy", "move"], default="copy", help="Modo de operación")
    parser.add_argument("--dry-run", type=str, default="true", choices=["true", "false"], help="Simular proceso")
    parser.add_argument("--threads", type=int, help="Número de hilos para procesamiento (sobrescribe config.yml)")
    parser.add_argument("--enable-content", action="store_true", help="Habilitar búsqueda en contenido")
    parser.add_argument("--enable-ocr", action="store_true", help="Habilitar OCR")
    parser.add_argument("--enable-users", action="store_true", help="Habilitar organización por usuario")
    parser.add_argument("--log", default="proceso.log", help="Archivo de log")
    parser.add_argument("--csv", default="resultados.csv", help="Archivo CSV de resultados")
    parser.add_argument("--verbose", action="store_true", help="Logs detallados")
    
    args = parser.parse_args()
    
    origen_path = Path(args.origen)
    destino_path = Path(args.destino)
    
    if not origen_path.exists():
        origen_path.mkdir(parents=True, exist_ok=True)
        print(f"Creada carpeta origen por defecto: {origen_path}")
    
    if not origen_path.is_dir():
        print(f"Error: La ruta origen no es una carpeta: {origen_path}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Actualizar configuración con argumentos de línea de comandos
    # SOLO si se especifican explícitamente
    config_updates = {
        "default_mode": args.mode,
        "dry_run": args.dry_run.lower() == "true",
        "enable_content_search": args.enable_content,
        "enable_ocr": args.enable_ocr,
        "enable_user_organization": args.enable_users,
        "log_file": args.log,
        "csv_file": args.csv,
        "verbose": args.verbose,
    }

    # Solo actualizar num_threads si se especifica explícitamente
    if args.threads is not None:
        config_updates["num_threads"] = args.threads

    config.update(config_updates)
    
    # Verificar dependencias opcionales
    if config["enable_content_search"]:
        missing_deps = []
        if not PDF_AVAILABLE:
            missing_deps.append("pdfplumber")
        if not DOCX_AVAILABLE:
            missing_deps.append("python-docx")
        if not XLSX_AVAILABLE:
            missing_deps.append("openpyxl")
        
        if missing_deps:
            print(f"Error: Para habilitar búsqueda en contenido, instale: {', '.join(missing_deps)}")
            sys.exit(1)
    
    if config["enable_ocr"] and not OCR_AVAILABLE:
        print("Error: Para habilitar OCR, instale: pip install Pillow pytesseract")
        print("También debe instalar Tesseract en su sistema")
        sys.exit(1)
    
    organizer = DocumentOrganizer(config)
    
    try:
        organizer.organize_documents(origen_path, destino_path)
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()