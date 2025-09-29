# Sistema de Penalizaciones Cruzadas

## Descripción
Se ha implementado un sistema de penalizaciones cruzadas para evitar confusiones entre categorías que comparten keywords similares.

## Penalizaciones Implementadas

### 1. Categoría 03 (Certificados de Estudios) vs Categoría 02 (Hoja de Vida)
**Problema**: Los certificados de estudios a veces contienen keywords de "hoja de vida" y se clasifican incorrectamente.

**Solución**: 
- Si un archivo tiene puntuación en Categoría 03 pero contiene keywords de Categoría 02, se aplica una penalización fuerte
- Keywords de penalización: `["hoja de vida", "formato hoja de vida unica", "cv", "vida unica", "hv", "curriculum vitae"]`

**Penalizaciones aplicadas**:
- **Nombre de archivo**: -20 puntos (penalización fuerte)
- **Carpeta padre**: -15 puntos (penalización moderada)
- **Contenido**: -10 puntos (penalización leve)
- **OCR**: -10 puntos (penalización leve)

### 2. Categoría 06 (Afiliación ARL) vs Categoría 00 (Check List)
**Problema**: Los documentos de ARL a veces contienen keywords de "check list" y se clasifican incorrectamente.

**Solución**:
- Si un archivo tiene puntuación en Categoría 06 pero contiene keywords de Categoría 00, se aplica una penalización fuerte
- Keywords de penalización: `["check list", "lista chequeo", "formato check list", "expediente laboral", "lista de chequeo"]`

**Penalizaciones aplicadas**:
- **Nombre de archivo**: -20 puntos (penalización fuerte)
- **Carpeta padre**: -15 puntos (penalización moderada)
- **Contenido**: -10 puntos (penalización leve)
- **OCR**: -10 puntos (penalización leve)

## Funcionamiento

### Nivel 1: Penalización en `get_keyword_score()`
- Se aplica directamente a las keywords individuales
- Puntuación reducida a 1 punto si contiene keywords de otras categorías
- Tipo de penalización: `"penalizado_por_hoja_vida"` o `"penalizado_por_check_list"`

### Nivel 2: Penalización en `classify_file()`
- Se aplica después de calcular todos los scores
- Penalización adicional basada en el análisis completo del archivo
- Se registra en los logs para seguimiento

## Logs y Seguimiento

El sistema registra todas las penalizaciones cruzadas aplicadas:
```
WARNING - Penalización cruzada aplicada: Categoría 03 penalizada por contener keywords de Hoja de Vida en archivo.pdf
WARNING - Penalización cruzada aplicada: Categoría 06 penalizada por contener keywords de Check List en documento.pdf
```

## Resultados en CSV

Las penalizaciones se registran en el campo `palabras_clave_encontradas` con etiquetas específicas:
- `penalizacion:contiene_hoja_vida`
- `penalizacion:contiene_check_list`
- `penalizacion_carpeta:contiene_hoja_vida`
- `penalizacion_carpeta:contiene_check_list`
- `penalizacion_contenido:contiene_hoja_vida`
- `penalizacion_contenido:contiene_check_list`
- `penalizacion_ocr:contiene_hoja_vida`
- `penalizacion_ocr:contiene_check_list`

## Configuración

Las penalizaciones están hardcodeadas en el código para mayor eficiencia, pero pueden ser modificadas fácilmente en la función `get_keyword_score()` y en las secciones de penalización cruzada de `classify_file()`.

## Beneficios

1. **Mayor precisión**: Reduce las clasificaciones incorrectas entre categorías similares
2. **Mejor organización**: Los documentos se clasifican en la categoría más apropiada
3. **Trazabilidad**: Todas las penalizaciones se registran para análisis posterior
4. **Flexibilidad**: El sistema puede ser extendido fácilmente para otras categorías
