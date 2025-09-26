"""
Organizador principal de documentos - Coordinador del sistema.
"""

import hashlib
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

from tqdm import tqdm

from .content_extractor import ContentExtractor
from .file_classifier import FileClassifier
from .ocr_processor import OCRProcessor
from ..utils.constants import SUPPORTED_EXTENSIONS, INVALID_FILENAME_CHARS
from ..utils.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class DocumentOrganizer:
    """Coordinador principal del sistema de organización de documentos."""
    
    def __init__(self, config: Dict, capabilities: Dict[str, bool]):
        self.config = config
        self.capabilities = capabilities
        
        # Inicializar componentes
        self.content_extractor = ContentExtractor(capabilities, config)
        self.ocr_processor = OCRProcessor(capabilities, config)
        self.file_classifier = FileClassifier(config, self.content_extractor, self.ocr_processor)
        self.report_generator = ReportGenerator(config)
        
        # Estado del procesamiento
        self.processed_hashes: Set[str] = set()
        self.lock = threading.Lock()
        
        # Configurar logging
        self._setup_logger()
        
        logger.info("DocumentOrganizer inicializado")
        logger.info(f"Capabilities: {list(k for k, v in capabilities.items() if v)}")
    
    def _setup_logger(self) -> None:
        """Configura el sistema de logging."""
        log_level = logging.DEBUG if self.config.get("verbose", False) else logging.INFO
        
        # Configurar logger principal
        main_logger = logging.getLogger("organizador_documentos")
        main_logger.setLevel(log_level)
        
        # Limpiar handlers existentes
        for handler in main_logger.handlers[:]:
            main_logger.removeHandler(handler)
        
        # Handler para archivo
        log_file = self.config.get("log_file", "proceso.log")
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
        except ImportError:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
        
        file_handler.setLevel(logging.DEBUG)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(funcName)s:%(lineno)d - %(message)s"
        )
        simple_formatter = logging.Formatter("%(levelname)s - %(message)s")
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        main_logger.addHandler(file_handler)
        main_logger.addHandler(console_handler)
    
    def organize_documents(self, origen_path: Path, destino_path: Path) -> None:
        """
        Organiza todos los documentos de la carpeta origen.
        
        Args:
            origen_path: Ruta de la carpeta origen
            destino_path: Ruta de la carpeta destino
        """
        start_time = time.time()
        
        logger.info("Iniciando organización de documentos")
        logger.info(f"Origen: {origen_path}")
        logger.info(f"Destino: {destino_path}")
        logger.info(f"Modo: {'DRY RUN' if self.config.get('dry_run', True) else self.config.get('default_mode', 'copy').upper()}")
        
        # Crear carpeta destino
        destino_path.mkdir(parents=True, exist_ok=True)
        
        # Encontrar archivos
        files = self._find_files(origen_path)
        if not files:
            logger.warning("No se encontraron archivos para procesar")
            return
        
        logger.info(f"Encontrados {len(files)} archivos para procesar")
        
        # Procesar archivos
        self._process_files_parallel(files, origen_path, destino_path)
        
        # Generar reportes
        processing_time = time.time() - start_time
        self.report_generator.set_stat("processing_time", processing_time)
        
        self._save_results_and_summary()
        
        logger.info(f"Procesamiento completado en {processing_time:.2f} segundos")
    
    def _find_files(self, origen_path: Path) -> List[Path]:
        """
        Encuentra todos los archivos soportados en la carpeta origen.
        
        Args:
            origen_path: Ruta de la carpeta origen
            
        Returns:
            Lista de rutas de archivos encontrados
        """
        files = []
        
        for file_path in origen_path.rglob("*"):
            if (file_path.is_file() and 
                file_path.suffix.lower() in SUPPORTED_EXTENSIONS):
                files.append(file_path)
        
        return files
    
    def _process_files_parallel(self, files: List[Path], origen_path: Path, destino_path: Path) -> None:
        """
        Procesa archivos en paralelo usando ThreadPoolExecutor.
        
        Args:
            files: Lista de archivos a procesar
            origen_path: Ruta de origen
            destino_path: Ruta de destino
        """
        num_threads = self.config.get("num_threads", 4)
        logger.info(f"Usando {num_threads} hilos para procesamiento")
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            with tqdm(total=len(files), desc="Procesando archivos") as pbar:
                # Enviar tareas
                future_to_file = {
                    executor.submit(self._process_single_file, file_path, origen_path, destino_path): file_path
                    for file_path in files
                }
                
                # Recoger resultados
                for future in as_completed(future_to_file):
                    try:
                        result = future.result()
                        with self.lock:
                            self.report_generator.add_result(result)
                    except Exception as e:
                        file_path = future_to_file[future]
                        logger.error(f"Error procesando {file_path}: {e}")
                        
                        # Crear resultado de error
                        error_result = self._create_error_result(file_path, origen_path, str(e))
                        with self.lock:
                            self.report_generator.add_result(error_result)
                    
                    pbar.update(1)
    
    def _process_single_file(self, file_path: Path, origen_path: Path, destino_path: Path) -> Dict:
        """
        Procesa un archivo individual.
        
        Args:
            file_path: Ruta del archivo
            origen_path: Ruta de origen
            destino_path: Ruta de destino
            
        Returns:
            Diccionario con resultado del procesamiento
        """
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
            # Calcular hash del archivo
            file_hash = self._calculate_file_hash(file_path)
            result["hash_sha256"] = file_hash
            
            # Verificar duplicados
            with self.lock:
                if file_hash in self.processed_hashes:
                    return self._handle_duplicate_file(file_path, destino_path, result)
                self.processed_hashes.add(file_hash)
            
            # Clasificar archivo
            category_id, reason, score, keywords = self.file_classifier.classify_file(file_path, origen_path)
            
            result["categoria_asignada"] = category_id
            result["razon_decision"] = reason
            result["puntuacion"] = score
            result["palabras_clave_encontradas"] = "; ".join(keywords)
            
            # Detectar usuario
            user_name = None
            if self.config.get("enable_user_organization", False):
                user_name = self.file_classifier.extract_user_from_folder(file_path, origen_path)
                if user_name:
                    self.report_generator.increment_stat("users_created")
            
            # Generar ruta de destino
            dest_path = self._generate_destination_path(file_path, category_id, destino_path, user_name)
            result["ruta_destino"] = str(dest_path.relative_to(destino_path))
            result["estado"] = "Clasificado"
            
            # Mover/copiar archivo si no es dry run
            if not self.config.get("dry_run", True):
                self._move_or_copy_file(file_path, dest_path)
            
            # Log del resultado
            if category_id == "Pendientes_Revisar":
                logger.warning(f"Archivo pendiente: {file_path.name} -> {reason}")
            else:
                logger.info(f"Archivo procesado: {file_path.name} -> {category_id} ({reason})")
            
        except Exception as e:
            result["estado"] = "Error"
            result["mensaje_error"] = str(e)
            logger.error(f"Error procesando {file_path}: {e}")
        
        return result
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcula hash SHA256 del archivo."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculando hash para {file_path}: {e}")
            return ""
    
    def _handle_duplicate_file(self, file_path: Path, destino_path: Path, result: Dict) -> Dict:
        """Maneja archivos duplicados."""
        result["estado"] = "Duplicado"
        result["categoria_asignada"] = "Duplicados"
        
        # Crear carpeta de duplicados
        duplicados_folder = destino_path / "Duplicados"
        duplicados_folder.mkdir(exist_ok=True)
        
        # Generar nombre único
        dest_path = self._get_unique_filename(duplicados_folder, file_path.name, "_dup")
        result["ruta_destino"] = str(dest_path.relative_to(destino_path))
        
        # Mover archivo si no es dry run
        if not self.config.get("dry_run", True):
            self._move_or_copy_file(file_path, dest_path)
        
        logger.info(f"Archivo duplicado: {file_path.name}")
        return result
    
    def _generate_destination_path(self, original_path: Path, category_id: str, 
                                 destino_path: Path, user_name: Optional[str] = None) -> Path:
        """
        Genera ruta de destino manteniendo el nombre original del archivo.
        
        Args:
            original_path: Ruta original del archivo
            category_id: ID de la categoría
            destino_path: Ruta base de destino
            user_name: Nombre del usuario (opcional)
            
        Returns:
            Ruta completa de destino
        """
        # Determinar carpeta base
        if user_name and self.config.get("enable_user_organization", False):
            base_folder = destino_path / user_name
        else:
            base_folder = destino_path
        
        # Crear carpeta de categoría
        if category_id == "Pendientes_Revisar":
            category_folder = base_folder / "Pendientes_Revisar"
        else:
            category_name = self.config["categories"].get(category_id, category_id)
            category_folder = base_folder / category_name
        
        category_folder.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre de archivo único
        original_filename = original_path.name
        return self._get_unique_filename(category_folder, original_filename)
    
    def _get_unique_filename(self, folder: Path, filename: str, suffix: str = "") -> Path:
        """
        Genera un nombre de archivo único en la carpeta especificada.
        
        Args:
            folder: Carpeta destino
            filename: Nombre original del archivo
            suffix: Sufijo adicional (opcional)
            
        Returns:
            Ruta completa con nombre único
        """
        clean_filename = self._clean_filename(filename)
        dest_path = folder / clean_filename
        
        if not dest_path.exists():
            return dest_path
        
        # Generar nombre único con contador
        name_part, ext = os.path.splitext(clean_filename)
        counter = 1
        
        while dest_path.exists():
            new_filename = f"{name_part}{suffix}_{counter}{ext}"
            dest_path = folder / new_filename
            counter += 1
        
        return dest_path
    
    def _clean_filename(self, filename: str) -> str:
        """Limpia nombre de archivo eliminando caracteres inválidos."""
        clean_name = filename
        
        for char in INVALID_FILENAME_CHARS:
            clean_name = clean_name.replace(char, "_")
        
        # Limitar longitud
        max_length = self.config.get("max_filename_length", 100)
        if len(clean_name) > max_length:
            name_part, ext = os.path.splitext(clean_name)
            clean_name = name_part[:max_length - len(ext)] + ext
        
        return clean_name
    
    def _move_or_copy_file(self, source: Path, destination: Path) -> None:
        """Mueve o copia archivo según configuración."""
        try:
            if self.config.get("default_mode", "copy") == "move":
                shutil.move(str(source), str(destination))
            else:
                shutil.copy2(str(source), str(destination))
        except Exception as e:
            logger.error(f"Error moviendo/copiando {source} -> {destination}: {e}")
            raise
    
    def _create_error_result(self, file_path: Path, origen_path: Path, error_msg: str) -> Dict:
        """Crea resultado de error para un archivo."""
        return {
            "timestamp": datetime.now().isoformat(),
            "ruta_original": str(file_path.relative_to(origen_path)),
            "hash_sha256": "",
            "categoria_asignada": "",
            "razon_decision": "",
            "puntuacion": 0,
            "palabras_clave_encontradas": "",
            "ruta_destino": "",
            "estado": "Error",
            "mensaje_error": error_msg,
        }
    
    def _save_results_and_summary(self) -> None:
        """Guarda resultados CSV y muestra resumen."""
        # Guardar CSV
        csv_saved = self.report_generator.save_results_csv()
        
        # Mostrar resumen
        self.report_generator.print_summary()
        
        # Estadísticas adicionales de componentes
        self._log_component_stats()
        
        if csv_saved:
            logger.info("Procesamiento completado exitosamente")
        else:
            logger.warning("Procesamiento completado con errores en el guardado")
    
    def _log_component_stats(self) -> None:
        """Registra estadísticas de los componentes."""
        # Estadísticas del clasificador
        classifier_stats = self.file_classifier.get_stats()
        logger.info(f"Estadísticas del clasificador: {classifier_stats}")
        
        # Estadísticas del OCR
        if self.capabilities.get('ocr_processing'):
            ocr_stats = self.ocr_processor.get_stats()
            logger.info(f"Estadísticas de OCR: {ocr_stats}")
            
            # Actualizar estadísticas del reporte
            self.report_generator.set_stat("cache_hits", ocr_stats.get("cache_hits", 0))
            self.report_generator.set_stat("cache_misses", ocr_stats.get("cache_misses", 0))
    
    def get_processing_stats(self) -> Dict:
        """Retorna estadísticas completas del procesamiento."""
        stats = self.report_generator.get_stats()
        
        # Añadir estadísticas de componentes
        stats["classifier"] = self.file_classifier.get_stats()
        
        if self.capabilities.get('ocr_processing'):
            stats["ocr"] = self.ocr_processor.get_stats()
        
        return stats