"""
Setup script para el organizador de documentos.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Leer requirements
requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="organizador-documentos",
    version="2.0.0",
    author="Document Organizer Team",
    author_email="",
    description="Sistema de clasificación automática de documentos usando OCR y análisis de contenido",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/organizador-documentos",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Office Suites",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=[
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "full": requirements,
        "content": [
            "pdfplumber>=0.7.0",
            "python-docx>=0.8.11",
            "openpyxl>=3.0.10",
        ],
        "ocr": [
            "Pillow>=9.0.0",
            "pytesseract>=0.3.10",
            "pdf2image>=3.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "organizador-documentos=organizador_documentos.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "organizador_documentos": [
            "config/*.yml",
        ],
    },
)