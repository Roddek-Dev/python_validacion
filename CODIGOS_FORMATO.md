# Sistema de Códigos de Formato - Máxima Prioridad

## Descripción
Se ha implementado un sistema de detección automática de códigos de formato específicos que tienen la máxima prioridad en la clasificación de documentos.

## Códigos de Formato Configurados

| Código | Categoría | Descripción |
|--------|-----------|-------------|
| `GAF-JTH-PD-01-FT-04` | 00 | Formato Check List |
| `GRH-PD-02-FT-07` | 01 | Requisición del Personal |
| `GRH-PD-02-FT-02` | 02 | Formato Hoja de Vida Unica |
| `GAF-JTH-PD-02-FT-11` | 16 | Certificación Bancaria |
| `GAF-JTH-PD-Ol-FT-03` | 19 | Constancia Inducción Corporativa |
| `GTE-PD-05-FT-01` | 21 | Autorización Tratamiento Datos Personales |
| `GAF-JTH-PD-02-FT-07` | 23 | Evaluación Periodo de Prueba |
| `GAF-JTH-SST-PD-07-FT-02` | 24 | Examen Médico Periódico |
| `GAF-JTH-PD-02-FT-02` | 25 | Permisos |
| `GAF-JTH-PD-02-FT-07` | 28 | Vacaciones |
| `GAF-JTH-PD-02-FT-08` | 32 | Formatos Deducción RTE FTE |
| `GAF-JTH-PD-01-FT-08` | 39 | Validaciones |

## Sistema de Puntuación

### Niveles de Prioridad:

1. **Códigos de Formato en Nombre de Archivo**: **75 puntos** (MÁXIMA PRIORIDAD)
2. **Códigos de Formato en Contenido**: **50 puntos** (ALTA PRIORIDAD)
3. **Códigos de Formato en OCR**: **50 puntos** (ALTA PRIORIDAD)
4. **Keywords críticas específicas**: **30 puntos**
5. **Keywords normales**: **10 puntos**
6. **Keywords genéricas**: **5 puntos**

### Detección Automática:

El sistema detecta automáticamente estos códigos en:
- **Nombre de archivo**: Máxima prioridad (75 puntos)
- **Contenido del documento**: Alta prioridad (50 puntos)
- **Texto extraído por OCR**: Alta prioridad (50 puntos)

## Funcionamiento

### 1. Detección en Nombre de Archivo
```python
# Si el archivo se llama: "GAF-JTH-PD-01-FT-04 - Check List.pdf"
# El sistema automáticamente:
# - Asigna 75 puntos a la categoría 00
# - Registra: "código_formato:GAF-JTH-PD-01-FT-04"
# - Log: "Código de formato detectado: GAF-JTH-PD-01-FT-04 -> Categoría 00"
```

### 2. Detección en Contenido
```python
# Si el contenido contiene: "GAF-JTH-PD-02-FT-07"
# El sistema automáticamente:
# - Asigna 50 puntos a la categoría 01
# - Registra: "código_formato_contenido:GAF-JTH-PD-02-FT-07"
# - Log: "Código de formato detectado en contenido: GAF-JTH-PD-02-FT-07 -> Categoría 01"
```

### 3. Detección en OCR
```python
# Si el OCR detecta: "GAF-JTH-SST-PD-07-FT-02"
# El sistema automáticamente:
# - Asigna 50 puntos a la categoría 24
# - Registra: "código_formato_ocr:GAF-JTH-SST-PD-07-FT-02"
# - Log: "Código de formato detectado en OCR: GAF-JTH-SST-PD-07-FT-02 -> Categoría 24"
```

## Ventajas del Sistema

1. **Clasificación Automática**: Los códigos de formato garantizan la clasificación correcta
2. **Máxima Precisión**: 75 puntos en nombre de archivo supera cualquier otra keyword
3. **Detección Múltiple**: Funciona en nombre, contenido y OCR
4. **Logs Detallados**: Registra cada detección para seguimiento
5. **Reducción de Pendientes**: Minimiza archivos sin clasificar

## Logs y Seguimiento

### Logs de Detección:
```
INFO - Código de formato detectado: GAF-JTH-PD-01-FT-04 -> Categoría 00 en archivo.pdf
INFO - Código de formato detectado en contenido: GAF-JTH-PD-02-FT-07 -> Categoría 01 en documento.pdf
INFO - Código de formato detectado en OCR: GAF-JTH-SST-PD-07-FT-02 -> Categoría 24 en imagen.pdf
```

### Registro en CSV:
- `código_formato:GAF-JTH-PD-01-FT-04`
- `código_formato_contenido:GAF-JTH-PD-02-FT-07`
- `código_formato_ocr:GAF-JTH-SST-PD-07-FT-02`

## Configuración

Los códigos están hardcodeados en el sistema para máxima eficiencia, pero pueden ser modificados fácilmente en:
- `get_keyword_score()`: Para keywords específicas
- `classify_file()`: Para detección automática

## Resultado Esperado

Con este sistema, los archivos que contengan códigos de formato específicos serán clasificados automáticamente en la categoría correcta, reduciendo significativamente el número de archivos pendientes de revisar.
