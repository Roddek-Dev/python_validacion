"""
Generador de reportes y estadísticas del procesamiento.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .constants import CLASSIFICATION_REASONS

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Genera reportes CSV, estadísticas y logs del procesamiento."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results: List[Dict] = []
        self.stats = {
            "total_files": 0,
            "classified": 0,
            "duplicates": 0,
            "pending": 0,
            "errors": 0,
            "ocr_used": 0,
            "content_analysis_used": 0,
            "users_created": 0,
            "processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
    
    def add_result(self, result: Dict) -> None:
        """Añade un resultado de procesamiento."""
        self.results.append(result)
        self._update_stats(result)
    
    def _update_stats(self, result: Dict) -> None:
        """Actualiza estadísticas basadas en un resultado."""
        estado = result.get("estado", "")
        
        if estado == "Clasificado":
            if result.get("categoria_asignada") == "Pendientes_Revisar":
                self.stats["pending"] += 1
            else:
                self.stats["classified"] += 1
        elif estado == "Duplicado":
            self.stats["duplicates"] += 1
        elif estado == "Error":
            self.stats["errors"] += 1
        
        # Contar uso de OCR y análisis de contenido
        keywords = result.get("palabras_clave_encontradas", "")
        if "ocr:" in keywords:
            self.stats["ocr_used"] += 1
        if "content:" in keywords:
            self.stats["content_analysis_used"] += 1
    
    def increment_stat(self, stat_name: str, value: int = 1) -> None:
        """Incrementa una estadística específica."""
        if stat_name in self.stats:
            self.stats[stat_name] += value
    
    def set_stat(self, stat_name: str, value: Any) -> None:
        """Establece el valor de una estadística."""
        self.stats[stat_name] = value
    
    def save_results_csv(self, csv_file: Optional[str] = None) -> bool:
        """
        Guarda resultados en archivo CSV.
        
        Args:
            csv_file: Ruta del archivo CSV (opcional)
            
        Returns:
            True si se guardó exitosamente
        """
        if not self.results:
            logger.warning("No hay resultados para guardar")
            return False
        
        csv_path = csv_file or self.config.get("csv_file", "resultados.csv")
        
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = self.results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            
            logger.info(f"Resultados guardados en: {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando CSV: {e}")
            return False
    
    def generate_summary_report(self) -> str:
        """Genera reporte de resumen textual."""
        if not self.results:
            return "No hay resultados para mostrar."
        
        total_files = len(self.results)
        self.stats["total_files"] = total_files
        
        # Agrupar por estado
        estados = {}
        for result in self.results:
            estado = result["estado"]
            estados[estado] = estados.get(estado, 0) + 1
        
        # Agrupar por categoría
        categorias = {}
        pendientes_por_razon = {}
        
        for result in self.results:
            if result["estado"] == "Clasificado":
                categoria = result["categoria_asignada"]
                categorias[categoria] = categorias.get(categoria, 0) + 1
                
                if categoria == "Pendientes_Revisar":
                    razon = result["razon_decision"]
                    pendientes_por_razon[razon] = pendientes_por_razon.get(razon, 0) + 1
        
        # Generar reporte
        report_lines = [
            "=" * 60,
            "RESUMEN DE PROCESAMIENTO",
            "=" * 60,
            f"Total de archivos procesados: {total_files}",
            f"Modo de ejecución: {'DRY RUN (simulación)' if self.config.get('dry_run', True) else 'REAL'}",
            f"Tiempo de procesamiento: {self.stats['processing_time']:.2f} segundos",
            "",
            "Por estado:"
        ]
        
        for estado, count in estados.items():
            report_lines.append(f"  {estado}: {count}")
        
        if categorias:
            report_lines.extend(["", "Por categoría:"])
            categories_config = self.config.get("categories", {})
            
            for categoria, count in sorted(categorias.items()):
                if categoria == "Pendientes_Revisar":
                    report_lines.append(f"  {categoria}: {count}")
                else:
                    category_name = categories_config.get(categoria, categoria)
                    report_lines.append(f"  {category_name}: {count}")
        
        if pendientes_por_razon:
            report_lines.extend(["", "Archivos pendientes de revisar por razón:"])
            for razon, count in pendientes_por_razon.items():
                razon_desc = CLASSIFICATION_REASONS.get(razon, razon)
                report_lines.append(f"  {razon_desc}: {count}")
        
        # Estadísticas adicionales
        report_lines.extend([
            "",
            f"Archivos duplicados encontrados: {self.stats['duplicates']}",
            f"Archivos pendientes de revisar: {categorias.get('Pendientes_Revisar', 0)}",
            f"Archivos con errores: {self.stats['errors']}",
            "",
            "Estadísticas de procesamiento:",
            f"  Análisis de contenido usado: {self.stats['content_analysis_used']} archivos",
            f"  OCR usado: {self.stats['ocr_used']} archivos",
            f"  Carpetas de usuario creadas: {self.stats['users_created']}",
        ])
        
        # Estadísticas de cache si están disponibles
        if self.stats['cache_hits'] > 0 or self.stats['cache_misses'] > 0:
            total_cache_ops = self.stats['cache_hits'] + self.stats['cache_misses']
            hit_rate = (self.stats['cache_hits'] / total_cache_ops) * 100 if total_cache_ops > 0 else 0
            report_lines.extend([
                f"  Cache hits: {self.stats['cache_hits']}",
                f"  Cache misses: {self.stats['cache_misses']}",
                f"  Cache hit rate: {hit_rate:.1f}%",
            ])
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def print_summary(self) -> None:
        """Imprime resumen del procesamiento."""
        summary = self.generate_summary_report()
        print(f"\n{summary}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas actuales."""
        return self.stats.copy()
    
    def export_detailed_report(self, output_path: str) -> bool:
        """
        Exporta reporte detallado a archivo.
        
        Args:
            output_path: Ruta del archivo de salida
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Reporte detallado - {datetime.now().isoformat()}\n")
                f.write(self.generate_summary_report())
                f.write("\n\n")
                
                # Detalles por archivo
                f.write("DETALLES POR ARCHIVO:\n")
                f.write("-" * 40 + "\n")
                
                for result in self.results:
                    f.write(f"Archivo: {result['ruta_original']}\n")
                    f.write(f"  Estado: {result['estado']}\n")
                    f.write(f"  Categoría: {result['categoria_asignada']}\n")
                    f.write(f"  Puntuación: {result['puntuacion']}\n")
                    f.write(f"  Razón: {result['razon_decision']}\n")
                    if result['palabras_clave_encontradas']:
                        f.write(f"  Keywords: {result['palabras_clave_encontradas']}\n")
                    f.write("\n")
            
            logger.info(f"Reporte detallado exportado a: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando reporte detallado: {e}")
            return False