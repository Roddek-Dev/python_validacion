# 📋 PENDIENTES DE MEJORAS - ORGANIZADOR DE DOCUMENTOS

## 🚨 **PROBLEMAS CRÍTICOS IDENTIFICADOS**

### **1. ARCHIVOS PENDIENTES DE REVISAR (16 archivos)**
Estos archivos no se están clasificando correctamente y necesitan ajustes:

#### **🔴 Alta Prioridad - Archivos con puntuación alta pero no clasificados:**
- `5. FOTOCOPIA CC CARLOS BAUTISTA.pdf` (45 pts) → **Debería ser categoría 10**
- `CARLOS ANDRES BAUTISTA GONZALÉZ CERT. HERRAMIENTAS GOOGLE 2025.pdf` (44 pts) → **Debería ser categoría 18**
- `26.4 CAMBIO CUENTA CARLOS BAUTISTA ENERO2024.pdf` (39 pts) → **Debería ser categoría 16**

#### **🟡 Media Prioridad - Archivos con puntuación media:**
- `19. FP CARLOS BAUTISTA.pdf` (24 pts) → **Debería ser categoría 18** (Tarjeta Profesional)
- `26.2 Concepto CARLOS ANDRES BAUTISTA GONZALEZ.PDF` (13 pts) → **Revisar categoría**
- `26.5.1 Acta individual de grado Magíster en Inteligencia de Negocios.pdf` (13 pts) → **Debería ser categoría 03**

#### **🟢 Baja Prioridad - Archivos con puntuación baja:**
- `26.1 EVALUACIÓN PERIODO DE PRUEBA CARLOS BAUTISTA.pdf` (6 pts) → **Debería ser categoría 23**
- `0.1 GAF-JTH-PD-01-FT-08 Formato Análisis de formación y experiencia (1).xlsx` (5 pts) → **Debería ser categoría 23**
- `EVALUACION PERIODO DE PRUEBA CARLOS BAUTISTA.pdf` (6 pts) → **Debería ser categoría 23**

#### **❌ Sin coincidencias:**
- `10. CERT. BANCARIA CARLOS BAUTISTA.pdf` (0 pts) → **Debería ser categoría 16**

---

## 🎯 **CATEGORÍAS QUE NECESITAN MEJORAS**

### **📊 Categorías con muy pocos archivos (posibles problemas de detección):**

#### **🔴 Críticas (1 archivo cada una):**
- **Categoría 01** (Requisición Personal) - Solo 1 archivo
- **Categoría 02** (Hoja de Vida) - Solo 1 archivo  
- **Categoría 05** (EPS) - Solo 1 archivo
- **Categoría 07** (Caja de Compensación) - Solo 1 archivo
- **Categoría 09** (Contrato de Trabajo) - Solo 1 archivo
- **Categoría 10** (Documento de Identidad) - Solo 1 archivo
- **Categoría 12** (Antecedentes Disciplinarios PGN) - Solo 1 archivo
- **Categoría 14** (Antecedentes Judiciales PONAL) - Solo 1 archivo
- **Categoría 18** (Tarjeta Profesional) - Solo 1 archivo
- **Categoría 19** (Constancia Inducción) - Solo 1 archivo

#### **🟡 Revisar (2-3 archivos):**
- **Categoría 03** (Certificados de Estudios) - 2 archivos
- **Categoría 06** (ARL) - 2 archivos
- **Categoría 11** (Examen Médico) - 3 archivos

---

## 🔧 **MEJORAS ESPECÍFICAS RECOMENDADAS**

### **1. CATEGORÍA 10 (Documento de Identidad)**
- **Problema**: Archivo `5. FOTOCOPIA CC CARLOS BAUTISTA.pdf` no se clasifica (45 pts < 50)
- **Solución**: Reducir umbral a 45 puntos o mejorar detección de patrones

### **2. CATEGORÍA 16 (Certificación Bancaria)**
- **Problema**: `10. CERT. BANCARIA CARLOS BAUTISTA.pdf` no se detecta (0 pts)
- **Solución**: Agregar keywords: "certificacion bancaria", "cuenta bancaria", "certificado bancario"

### **3. CATEGORÍA 18 (Tarjeta Profesional)**
- **Problema**: `19. FP CARLOS BAUTISTA.pdf` no se clasifica (24 pts)
- **Solución**: Agregar keywords: "fp", "fondo de pensiones", "tarjeta profesional"

### **4. CATEGORÍA 23 (Otros Documentos de Ingreso)**
- **Problema**: Archivos de "EVALUACIÓN PERIODO DE PRUEBA" no se detectan
- **Solución**: Agregar keywords: "evaluacion periodo", "periodo prueba", "evaluacion ingreso"

### **5. CATEGORÍA 03 (Certificados de Estudios)**
- **Problema**: `26.5.1 Acta individual de grado Magíster...` no se clasifica
- **Solución**: Agregar keywords: "acta individual", "grado", "magister", "diploma"

---

## 📝 **PLAN DE ACCIÓN SUGERIDO**

### **Fase 1: Correcciones Críticas (1-2 días)**
1. Ajustar umbral categoría 10 de 50 a 45 puntos
2. Mejorar keywords categoría 16 (bancaria)
3. Mejorar keywords categoría 18 (tarjeta profesional)

### **Fase 2: Mejoras de Detección (2-3 días)**
4. Agregar keywords categoría 23 (evaluación periodo prueba)
5. Mejorar keywords categoría 03 (certificados estudios)
6. Revisar keywords categorías 01, 02, 05, 07, 09

### **Fase 3: Optimización (1 día)**
7. Ajustar umbrales de confianza por categoría
8. Mejorar sistema de penalizaciones
9. Pruebas exhaustivas

---

## 🧪 **ARCHIVOS DE PRUEBA RECOMENDADOS**

Para probar las mejoras, usar estos archivos específicos:
- `5. FOTOCOPIA CC CARLOS BAUTISTA.pdf` → Categoría 10
- `10. CERT. BANCARIA CARLOS BAUTISTA.pdf` → Categoría 16  
- `19. FP CARLOS BAUTISTA.pdf` → Categoría 18
- `26.1 EVALUACIÓN PERIODO DE PRUEBA CARLOS BAUTISTA.pdf` → Categoría 23
- `26.5.1 Acta individual de grado Magíster...` → Categoría 03

---

## 📊 **ESTADÍSTICAS ACTUALES**
- **Total archivos procesados**: 53
- **Archivos pendientes**: 16 (30.2%)
- **Archivos clasificados**: 37 (69.8%)
- **Categorías con problemas**: 10+ categorías
- **Archivos duplicados**: 1

---

*Archivo generado automáticamente el $(date)*
*Basado en análisis de resultados.csv*
