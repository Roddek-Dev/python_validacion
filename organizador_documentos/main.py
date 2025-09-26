#!/usr/bin/env python3
"""
Entry point para el organizador de documentos.

Uso:
    python -m organizador_documentos --origen /ruta/origen --destino /ruta/destino
    python -m organizador_documentos --enable-ocr --enable-content --verbose
"""

import sys
from pathlib import Path
from typing import Optional

from .core.config_manager import ConfigManager
from .core.document_organizer import DocumentOrganizer
from .utils.dependency_manager import DependencyManager


def main() -> int:
    """Función principal del script."""
    try:
        # Inicializar gestores
        config_manager = ConfigManager()
        config = config_manager.parse_arguments_and_load_config()
        
        # Verificar dependencias
        dep_manager = DependencyManager()
        missing_deps = dep_manager.check_required_dependencies(config)
        
        if missing_deps:
            print("Error: Dependencias faltantes:")
            for dep, reason in missing_deps.items():
                print(f"  - {dep}: {reason}")
            return 1
        
        # Validar rutas
        origen_path = Path(config["origen"])
        destino_path = Path(config["destino"])
        
        if not origen_path.exists():
            origen_path.mkdir(parents=True, exist_ok=True)
            print(f"Creada carpeta origen por defecto: {origen_path}")
        
        if not origen_path.is_dir():
            print(f"Error: La ruta origen no es una carpeta: {origen_path}")
            return 1
        
        # Inicializar y ejecutar organizador
        organizer = DocumentOrganizer(config, dep_manager.capabilities)
        organizer.organize_documents(origen_path, destino_path)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario")
        return 1
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())