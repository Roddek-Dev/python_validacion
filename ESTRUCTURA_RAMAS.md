# Estructura de Ramas del Proyecto

## Ramas Disponibles

### 🌟 `funcional` (Rama Principal de Trabajo)
- **Propósito**: Contiene la versión estable y funcional del organizador de documentos
- **Estado**: ✅ Funcionando correctamente
- **Uso**: Para desarrollo diario y funcionalidades que ya están probadas
- **Última actualización**: Versión alfa funcional

### 🧪 `experimental` (Rama de Experimentación)
- **Propósito**: Contiene la versión experimental con nuevas funcionalidades
- **Estado**: ⚠️ En desarrollo, puede tener errores
- **Uso**: Para probar nuevas características y mejoras
- **Nota**: Esta rama puede fallar o no funcionar completamente

### 📋 `main` (Rama Base)
- **Propósito**: Rama base del proyecto
- **Estado**: Versión original del proyecto
- **Uso**: Punto de referencia para el historial del proyecto

## Flujo de Trabajo Recomendado

1. **Desarrollo diario**: Trabajar en la rama `funcional`
2. **Nuevas características**: Desarrollar en la rama `experimental`
3. **Cuando experimental esté estable**: Fusionar a `funcional`
4. **Releases importantes**: Fusionar `funcional` a `main`

## Comandos Útiles

```bash
# Cambiar a la rama funcional
git checkout funcional

# Cambiar a la rama experimental
git checkout experimental

# Ver todas las ramas
git branch -a

# Crear una nueva rama desde funcional
git checkout funcional
git checkout -b nueva-funcionalidad

# Fusionar experimental a funcional (cuando esté lista)
git checkout funcional
git merge experimental
```

## Archivos Importantes

- `organizador_documentos.py`: Script principal
- `config.yml`: Configuración del sistema
- `PENDIENTES_MEJORAS.md`: Lista de mejoras pendientes
- `PENDIENTES_SIMPLE.txt`: Lista simple de tareas pendientes
