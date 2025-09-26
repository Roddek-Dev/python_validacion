"""
Script para crear archivos de prueba sintéticos.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_files():
    """Crea archivos de prueba para testing."""
    fixtures_dir = Path(__file__).parent
    test_files_dir = fixtures_dir / "test_files"
    test_files_dir.mkdir(exist_ok=True)
    
    # Crear archivo TXT simple
    txt_file = test_files_dir / "documento_prueba.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("Este es un documento de prueba con contenido de texto simple.\n")
        f.write("Contiene palabras clave como: hoja de vida, requisición, certificado.\n")
    
    # Crear imagen con texto para OCR
    create_test_image(test_files_dir / "imagen_con_texto.png")
    
    # Crear estructura de carpetas de usuario
    user_folder = test_files_dir / "CARLOS BAUTISTA - CC 1234567890"
    user_folder.mkdir(exist_ok=True)
    
    # Archivo en carpeta de usuario
    user_file = user_folder / "hoja_de_vida.txt"
    with open(user_file, 'w', encoding='utf-8') as f:
        f.write("Hoja de vida de Carlos Bautista\n")
        f.write("Documento de identidad: 1234567890\n")
    
    print(f"Archivos de prueba creados en: {test_files_dir}")

def create_test_image(image_path: Path):
    """Crea una imagen de prueba con texto."""
    # Crear imagen blanca
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Añadir texto
    try:
        # Intentar usar una fuente del sistema
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        # Usar fuente por defecto si no encuentra arial
        font = ImageFont.load_default()
    
    text_lines = [
        "DOCUMENTO DE IDENTIDAD",
        "CEDULA DE CIUDADANIA",
        "Número: 1234567890",
        "Nombre: CARLOS BAUTISTA"
    ]
    
    y_position = 20
    for line in text_lines:
        draw.text((20, y_position), line, fill='black', font=font)
        y_position += 30
    
    img.save(image_path)

if __name__ == "__main__":
    create_test_files()