#!/usr/bin/env python3
"""
Script de debug para verificar el funcionamiento de las keywords.
"""

import sys
import unicodedata
import re
from pathlib import Path

# Agregar el directorio del proyecto al path
sys.path.insert(0, str(Path(__file__).parent))

from organizador_documentos.core.file_classifier import FileClassifier
from organizador_documentos.core.config_manager import ConfigManager

def normalize_text(text: str) -> str:
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

def debug_keywords():
    """Debug de keywords."""
    # Cargar configuración
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print(f"Config keys: {list(config.keys())}")
    if "keywords" in config:
        print(f"Keywords disponibles: {len(config['keywords'])} categorías")
    else:
        print("ERROR: No hay keywords en la configuración!")
        return
    
    # Crear clasificador
    classifier = FileClassifier(config, None, None)
    
    # Archivos de prueba
    test_files = [
        "20. AFILIACION ARL CARLOS ANDRES BAUTISTA GONZALEZ.pdf",
        "5. FOTOCOPIA CC CARLOS BAUTISTA.pdf", 
        "2. Hoja de vida.pdf",
        "18. Afiliación EPS - Carlos Andrés Bautista González.pdf"
    ]
    
    print("=== DEBUG DE KEYWORDS ===\n")
    
    for filename in test_files:
        print(f"Archivo: {filename}")
        normalized = normalize_text(filename)
        print(f"Normalizado: '{normalized}'")
        
        # Probar cada categoría
        for cat_id, cat_keywords in config["keywords"].items():
            matches = []
            for keyword in cat_keywords:
                normalized_keyword = normalize_text(keyword)
                if normalized_keyword in normalized:
                    matches.append(keyword)
            
            if matches:
                print(f"  Categoría {cat_id}: {matches}")
        
        print()

if __name__ == "__main__":
    debug_keywords()
