"""
Gestor de configuración y argumentos de línea de comandos.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from ..utils.constants import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ConfigManager:
    """Gestiona la configuración del sistema y argumentos de línea de comandos."""
    
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
    
    def load_config_file(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Carga configuración desde archivo YAML.
        
        Args:
            config_path: Ruta al archivo de configuración (opcional)
            
        Returns:
            Diccionario con la configuración cargada
        """
        config_file = None
        
        if config_path:
            config_file = Path(config_path)
        else:
            # Buscar config.yml en directorio actual
            local_config = Path("config.yml")
            if local_config.exists():
                config_file = local_config
        
        if config_file and config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    file_config = yaml.safe_load(f) or {}
                    self.config.update(file_config)
                    logger.info(f"Configuración cargada desde: {config_file}")
            except Exception as e:
                logger.error(f"Error cargando configuración desde {config_file}: {e}")
                logger.info("Usando configuración por defecto")
        else:
            logger.info("Usando configuración por defecto")
        
        return self.config
    
    def parse_arguments(self) -> argparse.Namespace:
        """
        Parsea argumentos de línea de comandos.
        
        Returns:
            Namespace con argumentos parseados
        """
        parser = argparse.ArgumentParser(
            description="Organizador de Documentos - Automatización de clasificación de documentos",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Ejemplos de uso:
  python -m organizador_documentos --origen /ruta/origen --destino /ruta/destino
  python -m organizador_documentos --enable-ocr --enable-content --verbose
            """,
        )

        default_origen = str(Path.home() / "Escritorio" / "Documentos_Desordenados")
        default_destino = str(Path.home() / "Escritorio" / "Documentos_Organizados")

        parser.add_argument(
            "--origen", 
            required=False, 
            default=default_origen, 
            help="Ruta a la carpeta de origen"
        )
        parser.add_argument(
            "--destino", 
            required=False, 
            default=default_destino, 
            help="Ruta a la carpeta de destino"
        )
        parser.add_argument(
            "--config", 
            help="Ruta al archivo config.yml"
        )
        parser.add_argument(
            "--mode", 
            choices=["copy", "move"], 
            default="copy", 
            help="Modo de operación"
        )
        parser.add_argument(
            "--dry-run", 
            type=str, 
            default="true", 
            choices=["true", "false"], 
            help="Simular proceso"
        )
        parser.add_argument(
            "--threads", 
            type=int, 
            help="Número de hilos para procesamiento (sobrescribe config.yml)"
        )
        parser.add_argument(
            "--enable-content", 
            action="store_true", 
            help="Habilitar búsqueda en contenido"
        )
        parser.add_argument(
            "--enable-ocr", 
            action="store_true", 
            help="Habilitar OCR"
        )
        parser.add_argument(
            "--enable-users", 
            action="store_true", 
            help="Habilitar organización por usuario"
        )
        parser.add_argument(
            "--log", 
            default="proceso.log", 
            help="Archivo de log"
        )
        parser.add_argument(
            "--csv", 
            default="resultados.csv", 
            help="Archivo CSV de resultados"
        )
        parser.add_argument(
            "--verbose", 
            action="store_true", 
            help="Logs detallados"
        )
        
        return parser.parse_args()
    
    def merge_args_with_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Combina argumentos de línea de comandos con configuración.
        
        Args:
            args: Argumentos parseados
            
        Returns:
            Configuración final combinada
        """
        # Actualizar configuración con argumentos de línea de comandos
        config_updates = {
            "origen": args.origen,
            "destino": args.destino,
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

        self.config.update(config_updates)
        return self.config
    
    def parse_arguments_and_load_config(self) -> Dict[str, Any]:
        """
        Parsea argumentos y carga configuración en un solo paso.
        
        Returns:
            Configuración final combinada
        """
        args = self.parse_arguments()
        
        # Cargar configuración desde archivo primero
        self.load_config_file(args.config)
        
        # Luego combinar con argumentos de línea de comandos
        return self.merge_args_with_config(args)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Valida la configuración cargada.
        
        Args:
            config: Configuración a validar
            
        Returns:
            True si la configuración es válida
        """
        required_keys = [
            "categories", "keywords", "num_threads", 
            "confidence_threshold", "origen", "destino"
        ]
        
        for key in required_keys:
            if key not in config:
                logger.error(f"Clave requerida faltante en configuración: {key}")
                return False
        
        # Validar tipos
        if not isinstance(config["num_threads"], int) or config["num_threads"] < 1:
            logger.error("num_threads debe ser un entero positivo")
            return False
        
        if not isinstance(config["confidence_threshold"], (int, float)) or config["confidence_threshold"] < 0:
            logger.error("confidence_threshold debe ser un número no negativo")
            return False
        
        # Validar que existan categorías y keywords
        if not config.get("categories") or not config.get("keywords"):
            logger.error("Configuración debe incluir categories y keywords")
            return False
        
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """Retorna la configuración actual."""
        return self.config.copy()
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Actualiza la configuración con nuevos valores.
        
        Args:
            updates: Diccionario con actualizaciones
        """
        self.config.update(updates)
        logger.debug(f"Configuración actualizada con: {list(updates.keys())}")