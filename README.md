# Simulador de Gas Ideal en 2D

## Descripción del Proyecto

Este proyecto implementa un simulador computacional de un gas ideal confinado en una caja cuadrada en 2D. El modelo permite estudiar el comportamiento de las partículas, verificar la conservación de energía y validar la relación entre temperatura y velocidad según la teoría cinética de gases.

## Estructura del Proyecto

```
gas_ideal/
│
├── particula.py        # Clase Partícula con física de colisiones
├── simulacion.py       # Motor de simulación y análisis
├── test_gas.py         # Suite completa de pruebas unitarias
├── graficar.py         # Herramientas de visualización
└── README.md           # Esta documentación
```

## Características Principales

### 🔬 Física Implementada

- **Colisiones elásticas** entre partículas (conservación de momento y energía)
- **Colisiones con paredes** (rebotes perfectamente elásticos)
- **Distribución de Maxwell-Boltzmann** para velocidades iniciales
- **Teorema de equipartición** para cálculo de temperatura
- **Conservación de energía** y momento total

### 📊 Análisis y Verificación

- Verificación automática de conservación de energía
- Validación de la relación T-v: v_rms = √(3k_B·T/m)
- Estadísticas completas del sistema
- Comparación con predicciones teóricas

### 📈 Visualización

- Gráficas de evolución temporal de energía
- Relación temperatura-velocidad
- Distribuciones de velocidad vs Maxwell-Boltzmann
- Animaciones del movimiento de partículas

## Instalación de Dependencias

```bash
pip install numpy matplotlib
```

## Uso Rápido

### 1. Ejecutar Pruebas Unitarias

```python
python test_gas.py
```

Esto ejecutará todas las pruebas de validación física y numérica.

### 2. Simulación Básica

```python
from simulacion import SimulacionGasIdeal

# Crear simulación
sim = SimulacionGasIdeal(
    n_particulas=50,           # Número de partículas
    lado_caja=1e-6,            # Lado de la caja en metros (1 μm)
    temperatura_inicial=300    # Temperatura inicial en Kelvin
)

# Ejecutar simulación
sim.ejecutar(
    t_total=1e-10,    # Tiempo total: 0.1 nanosegundos
    dt=1e-13,         # Paso temporal: 0.1 picosegundos
    guardar_cada=10   # Guardar datos cada 10 pasos
)

# Obtener estadísticas
stats = sim.obtener_estadisticas()
print(f"Temperatura media: {stats['temperatura_media']:.2f} K")
print(f"Energía conservada: {stats['energia_variacion']:.4f}% variación")
```

### 3. Generar Visualizaciones

```python
from graficar import reporte_completo

# Generar reporte completo con todas las gráficas
reporte_completo(sim, guardar_figuras=True)
```

### 4. Crear Animación

```python
from graficar import animar_particulas
from simulacion import SimulacionGasIdeal

# Crear nueva simulación para animación
sim_anim = SimulacionGasIdeal(n_particulas=30, lado_caja=1e-6, temperatura_inicial=300)

# Crear animación
animar_particulas(sim_anim, n_frames=200, guardar=True, archivo='gas_ideal.gif')
```

## Ejemplo Completo de Análisis

```python
import numpy as np
from simulacion import SimulacionGasIdeal
from graficar import (graficar_conservacion_energia, 
                      graficar_temperatura_velocidad,
                      graficar_distribucion_velocidades)

# Crear simulación con parámetros realistas
# (similar a gas Helio a temperatura ambiente)
sim = SimulacionGasIdeal(
    n_particulas=100,
    lado_caja=1e-6,              # Caja de 1 μm
    masa_particula=6.64e-27,     # Masa del átomo de Helio
    temperatura_inicial=300       # Temperatura ambiente
)

# Ejecutar simulación suficientemente larga para equilibrio
print("Ejecutando simulación...")
sim.ejecutar(t_total=2e-10, dt=1e-13, guardar_cada=10)

# Verificar conservación de energía
conserva, variacion = sim.verificar_conservacion_energia()
print(f"\n¿Se conserva la energía? {conserva}")
print(f"Variación: {variacion:.4f}%")

# Verificar relación temperatura-velocidad
cumple, error = sim.verificar_relacion_temperatura_velocidad()
print(f"\n¿Cumple la relación T-v? {cumple}")
print(f"Error relativo: {error*100:.2f}%")

# Obtener estadísticas
stats = sim.obtener_estadisticas()
print(f"\nEstadísticas:")
print(f"  Temperatura media: {stats['temperatura_media']:.2f} K")
print(f"  v_rms simulada: {stats['v_rms_media']:.1f} m/s")
print(f"  v_rms teórica: {stats['v_rms_teorica']:.1f} m/s")

# Generar gráficas
graficar_conservacion_energia(sim, guardar=True)
graficar_temperatura_velocidad(sim, guardar=True)
graficar_distribucion_velocidades(sim, guardar=True)
```

## Fundamentos Teóricos

### Teoría Cinética de Gases

1. **Energía cinética media por partícula (2D)**:
   ```
   <E> = k_B · T
   ```
   donde k_B = 1.380649×10⁻²³ J/K es la constante de Boltzmann

2. **Velocidad cuadrática media (2D)**:
   ```
   v_rms = √(<v²>) = √(2k_B·T/m)
   ```
   Nota: En 3D la fórmula sería √(3k_B·T/m)

3. **Distribución de Maxwell-Boltzmann (2D)**:
   ```
   f(v) = (m/(k_B·T)) · v · exp(-m·v²/(2k_B·T))
   ```

### Colisiones Elásticas

- **Con paredes**: Inversión de la componente perpendicular de velocidad
- **Entre partículas**: Conservación de momento y energía cinética

## Parámetros Típicos

| Gas    | Masa (kg)    | T (K) | v_rms (m/s) |
|--------|--------------|-------|-------------|
| H₂     | 3.35×10⁻²⁷   | 300   | ~1927       |
| He     | 6.64×10⁻²⁷   | 300   | ~1363       |
| N₂     | 4.65×10⁻²⁶   | 300   | ~517        |
| Ar     | 6.63×10⁻²⁶   | 300   | ~433        |

## Validación del Modelo

Las pruebas unitarias verifican:

✅ **Conservación de energía**: < 1% de variación  
✅ **Conservación de momento**: Momento total ≈ 0  
✅ **Relación T-v**: Error < 5%  
✅ **Distribución de velocidades**: Sigue Maxwell-Boltzmann  
✅ **Confinamiento**: Partículas permanecen en la caja  

## Limitaciones del Modelo

- **2D**: Simulación en 2 dimensiones (más simple que gases reales 3D)
- **Gas ideal**: No hay fuerzas intermoleculares
- **Colisiones binarias**: Solo se consideran colisiones de 2 partículas
- **Paredes rígidas**: No se modela la interacción detallada con las paredes

## Extensiones Posibles

1. Añadir fuerzas intermoleculares (potencial de Lennard-Jones)
2. Simular en 3D
3. Paredes móviles para estudiar compresión/expansión
4. Partículas de diferentes masas (mezclas de gases)
5. Campos externos (gravedad, eléctricos)
6. Termalización con baño térmico

## Referencias

- Reif, F. "Fundamentals of Statistical and Thermal Physics"
- Landau & Lifshitz "Statistical Physics"
- Allen & Tildesley "Computer Simulation of Liquids"

## Autor

Proyecto desarrollado para el estudio computacional de sistemas termodinámicos.

## Licencia

MIT License - Libre uso con atribución.