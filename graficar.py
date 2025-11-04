import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from simulacion import SimulacionGasIdeal

def graficar_conservacion_energia(sim, guardar=False, archivo='energia.png'):
    """
    Grafica la evolución de la energía total en el tiempo.
    
    Args:
        sim: objeto SimulacionGasIdeal con historial
        guardar: si guardar la figura
        archivo: nombre del archivo
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Gráfica 1: Energía vs tiempo
    tiempo_ns = np.array(sim.tiempo) * 1e9  # Convertir a nanosegundos
    energia_J = np.array(sim.historial_energia)
    
    ax1.plot(tiempo_ns, energia_J, 'b-', linewidth=1.5, label='Energía total')
    ax1.axhline(y=np.mean(energia_J), color='r', linestyle='--', 
                label=f'Media: {np.mean(energia_J):.3e} J')
    ax1.set_xlabel('Tiempo (ns)', fontsize=12)
    ax1.set_ylabel('Energía Total (J)', fontsize=12)
    ax1.set_title('Conservación de Energía', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Variación relativa de energía
    energia_inicial = energia_J[0]
    variacion_relativa = (energia_J - energia_inicial) / energia_inicial * 100
    
    ax2.plot(tiempo_ns, variacion_relativa, 'g-', linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Tiempo (ns)', fontsize=12)
    ax2.set_ylabel('Variación Relativa (%)', fontsize=12)
    ax2.set_title('Variación de Energía Respecto al Valor Inicial', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Estadísticas
    stats_text = f'Variación máxima: {np.max(np.abs(variacion_relativa)):.3f}%\n'
    stats_text += f'Desviación estándar: {np.std(energia_J):.3e} J'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        print(f"Figura guardada como {archivo}")
    
    plt.show()


def graficar_temperatura_velocidad(sim, guardar=False, archivo='temperatura_velocidad.png'):
    """
    Grafica la relación entre temperatura y velocidad cuadrática media.
    
    Args:
        sim: objeto SimulacionGasIdeal con historial
        guardar: si guardar la figura
        archivo: nombre del archivo
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    tiempo_ns = np.array(sim.tiempo) * 1e9
    temperatura = np.array(sim.historial_temperatura)
    v_rms = np.array(sim.historial_velocidades)
    
    # Gráfica 1: Temperatura vs tiempo
    ax1.plot(tiempo_ns, temperatura, 'r-', linewidth=1.5, label='T simulada')
    ax1.axhline(y=np.mean(temperatura), color='darkred', linestyle='--',
                label=f'T media: {np.mean(temperatura):.1f} K')
    ax1.set_xlabel('Tiempo (ns)', fontsize=12)
    ax1.set_ylabel('Temperatura (K)', fontsize=12)
    ax1.set_title('Evolución de la Temperatura', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: v_rms vs temperatura
    ax2.scatter(temperatura, v_rms, c=tiempo_ns, cmap='viridis', s=20, alpha=0.6)
    
    # Relación teórica: v_rms = sqrt(2*k_B*T/m) para 2D
    T_range = np.linspace(temperatura.min(), temperatura.max(), 100)
    k_B = 1.380649e-23
    v_rms_teorica = np.sqrt(2 * k_B * T_range / sim.masa_particula)
    
    ax2.plot(T_range, v_rms_teorica, 'r--', linewidth=2, 
             label='Relación teórica (2D)\n$v_{rms} = \\sqrt{2k_BT/m}
    
    ax2.set_xlabel('Temperatura (K)', fontsize=12)
    ax2.set_ylabel('Velocidad RMS (m/s)', fontsize=12)
    ax2.set_title('Relación Temperatura-Velocidad', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Tiempo (ns)', fontsize=10)
    
    # Estadísticas
    v_rms_teorica_media = np.sqrt(3 * k_B * np.mean(temperatura) / sim.masa_particula)
    error = abs(np.mean(v_rms) - v_rms_teorica_media) / v_rms_teorica_media * 100
    
    stats_text = f'$v_{{rms}}$ simulada: {np.mean(v_rms):.1f} m/s\n'
    stats_text += f'$v_{{rms}}$ teórica: {v_rms_teorica_media:.1f} m/s\n'
    stats_text += f'Error relativo: {error:.2f}%'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        print(f"Figura guardada como {archivo}")
    
    plt.show()


def graficar_distribucion_velocidades(sim, guardar=False, archivo='distribucion_velocidades.png'):
    """
    Grafica la distribución de velocidades y compara con Maxwell-Boltzmann.
    
    Args:
        sim: objeto SimulacionGasIdeal
        guardar: si guardar la figura
        archivo: nombre del archivo
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Recolectar velocidades
    velocidades_x = [p.velocidad[0] for p in sim.particulas]
    velocidades_y = [p.velocidad[1] for p in sim.particulas]
    velocidades_magnitud = [np.linalg.norm(p.velocidad) for p in sim.particulas]
    
    # Gráfica 1: Distribución de componentes de velocidad
    ax1.hist(velocidades_x, bins=20, alpha=0.6, label='$v_x$', density=True, color='blue')
    ax1.hist(velocidades_y, bins=20, alpha=0.6, label='$v_y$', density=True, color='red')
    
    # Distribución teórica Gaussiana
    T = sim.temperatura()
    k_B = 1.380649e-23
    sigma = np.sqrt(k_B * T / sim.masa_particula)
    v_range = np.linspace(-3*sigma, 3*sigma, 1000)
    gauss = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-v_range**2/(2*sigma**2))
    
    ax1.plot(v_range, gauss, 'k--', linewidth=2, label='Gaussiana teórica')
    ax1.set_xlabel('Velocidad (m/s)', fontsize=12)
    ax1.set_ylabel('Densidad de Probabilidad', fontsize=12)
    ax1.set_title('Distribución de Componentes de Velocidad', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Distribución de magnitudes de velocidad (Maxwell 2D)
    ax2.hist(velocidades_magnitud, bins=20, alpha=0.7, density=True, color='green', label='Simulación')
    
    # Distribución de Maxwell 2D: f(v) = (m/(k_B*T)) * v * exp(-m*v²/(2*k_B*T))
    v_mag_range = np.linspace(0, max(velocidades_magnitud)*1.2, 1000)
    maxwell_2d = (sim.masa_particula/(k_B*T)) * v_mag_range * np.exp(-sim.masa_particula*v_mag_range**2/(2*k_B*T))
    
    ax2.plot(v_mag_range, maxwell_2d, 'r--', linewidth=2, label='Maxwell 2D teórica')
    ax2.set_xlabel('Magnitud de Velocidad (m/s)', fontsize=12)
    ax2.set_ylabel('Densidad de Probabilidad', fontsize=12)
    ax2.set_title('Distribución de Maxwell (2D)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        print(f"Figura guardada como {archivo}")
    
    plt.show()


def animar_particulas(sim, n_frames=200, intervalo=50, guardar=False, archivo='animacion.gif'):
    """
    Crea una animación de las partículas en movimiento.
    
    Args:
        sim: objeto SimulacionGasIdeal
        n_frames: número de frames
        intervalo: milisegundos entre frames
        guardar: si guardar la animación
        archivo: nombre del archivo
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Configurar ejes
    ax.set_xlim(sim.limites_x[0], sim.limites_x[1])
    ax.set_ylim(sim.limites_y[0], sim.limites_y[1])
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('Simulación de Gas Ideal en 2D', fontsize=14, fontweight='bold')
    
    # Dibujar caja
    caja = Rectangle((sim.limites_x[0], sim.limites_y[0]), 
                      sim.lado_caja, sim.lado_caja,
                      fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(caja)
    
    # Crear círculos para cada partícula
    circulos = []
    for p in sim.particulas:
        circulo = Circle(p.posicion, p.radio, color='blue', alpha=0.7)
        ax.add_patch(circulo)
        circulos.append(circulo)
    
    # Texto para información
    texto_info = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    dt = 1e-13  # Paso de tiempo
    
    def actualizar(frame):
        # Avanzar simulación
        for _ in range(10):  # Varios pasos por frame para acelerar
            sim.paso_temporal(dt)
        
        # Actualizar posiciones de círculos
        for circulo, particula in zip(circulos, sim.particulas):
            circulo.center = particula.posicion
        
        # Actualizar información
        E = sim.energia_total()
        T = sim.temperatura()
        v_rms = sim.velocidad_cuadratica_media()
        
        info = f'Tiempo: {sim.t_actual*1e9:.3f} ns\n'
        info += f'Energía: {E:.3e} J\n'
        info += f'Temperatura: {T:.1f} K\n'
        info += f'$v_{{rms}}$: {v_rms:.1f} m/s'
        texto_info.set_text(info)
        
        return circulos + [texto_info]
    
    anim = animation.FuncAnimation(fig, actualizar, frames=n_frames,
                                   interval=intervalo, blit=True, repeat=True)
    
    if guardar:
        print(f"Guardando animación como {archivo}...")
        anim.save(archivo, writer='pillow', fps=20, dpi=100)
        print(f"Animación guardada como {archivo}")
    
    plt.show()


def reporte_completo(sim, guardar_figuras=False):
    """
    Genera un reporte completo con todas las gráficas y estadísticas.
    
    Args:
        sim: objeto SimulacionGasIdeal con historial
        guardar_figuras: si guardar las figuras
    """
    print("=" * 70)
    print("REPORTE DE SIMULACIÓN DE GAS IDEAL")
    print("=" * 70)
    print()
    
    # Estadísticas
    stats = sim.obtener_estadisticas()
    
    print("PARÁMETROS DE SIMULACIÓN:")
    print(f"  Número de partículas: {sim.n_particulas}")
    print(f"  Lado de la caja: {sim.lado_caja*1e6:.2f} μm")
    print(f"  Masa de partícula: {sim.masa_particula:.3e} kg")
    print(f"  Tiempo simulado: {stats['tiempo_total']*1e9:.3f} ns")
    print()
    
    print("RESULTADOS:")
    print(f"  Energía media: {stats['energia_media']:.6e} J")
    print(f"  Desviación energía: {stats['energia_std']:.6e} J")
    print(f"  Variación energía: {stats['energia_variacion']:.4f}%")
    print()
    print(f"  Temperatura media: {stats['temperatura_media']:.2f} K")
    print(f"  Desviación temperatura: {stats['temperatura_std']:.2f} K")
    print()
    print(f"  v_rms simulada: {stats['v_rms_media']:.2f} m/s")
    print(f"  v_rms teórica: {stats['v_rms_teorica']:.2f} m/s")
    print(f"  Error relativo: {abs(stats['v_rms_media']-stats['v_rms_teorica'])/stats['v_rms_teorica']*100:.2f}%")
    print()
    
    # Verificaciones
    conserva, variacion = sim.verificar_conservacion_energia()
    print("VERIFICACIÓN DE FÍSICA:")
    print(f"  Conservación de energía: {'✓ SÍ' if conserva else '✗ NO'}")
    print(f"  Variación de energía: {variacion:.4f}%")
    
    cumple, error = sim.verificar_relacion_temperatura_velocidad()
    print(f"  Relación T-v correcta: {'✓ SÍ' if cumple else '✗ NO'}")
    print(f"  Error en relación T-v: {error*100:.2f}%")
    print()
    print("=" * 70)
    
    # Generar gráficas
    print("\nGenerando gráficas...")
    graficar_conservacion_energia(sim, guardar_figuras, 'conservacion_energia.png')
    graficar_temperatura_velocidad(sim, guardar_figuras, 'temperatura_velocidad.png')
    graficar_distribucion_velocidades(sim, guardar_figuras, 'distribucion_velocidades.png')


if __name__ == '__main__':
    print("Creando y ejecutando simulación...")
    
    # Crear simulación
    sim = SimulacionGasIdeal(n_particulas=50, lado_caja=1e-6, temperatura_inicial=300)
    
    # Ejecutar simulación
    sim.ejecutar(t_total=1e-10, dt=1e-13, guardar_cada=10)
    
    # Generar reporte
    reporte_completo(sim, guardar_figuras=True)
    
    # Opcional: crear animación (comentar si no se desea)
    # print("\nCreando animación...")
    # sim_anim = SimulacionGasIdeal(n_particulas=30, lado_caja=1e-6, temperatura_inicial=300)
    # animar_particulas(sim_anim, n_frames=100, guardar=True)
)
    
    ax2.set_xlabel('Temperatura (K)', fontsize=12)
    ax2.set_ylabel('Velocidad RMS (m/s)', fontsize=12)
    ax2.set_title('Relación Temperatura-Velocidad', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Tiempo (ns)', fontsize=10)
    
    # Estadísticas
    v_rms_teorica_media = np.sqrt(3 * k_B * np.mean(temperatura) / sim.masa_particula)
    error = abs(np.mean(v_rms) - v_rms_teorica_media) / v_rms_teorica_media * 100
    
    stats_text = f'$v_{{rms}}$ simulada: {np.mean(v_rms):.1f} m/s\n'
    stats_text += f'$v_{{rms}}$ teórica: {v_rms_teorica_media:.1f} m/s\n'
    stats_text += f'Error relativo: {error:.2f}%'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        print(f"Figura guardada como {archivo}")
    
    plt.show()


def graficar_distribucion_velocidades(sim, guardar=False, archivo='distribucion_velocidades.png'):
    """
    Grafica la distribución de velocidades y compara con Maxwell-Boltzmann.
    
    Args:
        sim: objeto SimulacionGasIdeal
        guardar: si guardar la figura
        archivo: nombre del archivo
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Recolectar velocidades
    velocidades_x = [p.velocidad[0] for p in sim.particulas]
    velocidades_y = [p.velocidad[1] for p in sim.particulas]
    velocidades_magnitud = [np.linalg.norm(p.velocidad) for p in sim.particulas]
    
    # Gráfica 1: Distribución de componentes de velocidad
    ax1.hist(velocidades_x, bins=20, alpha=0.6, label='$v_x$', density=True, color='blue')
    ax1.hist(velocidades_y, bins=20, alpha=0.6, label='$v_y$', density=True, color='red')
    
    # Distribución teórica Gaussiana
    T = sim.temperatura()
    k_B = 1.380649e-23
    sigma = np.sqrt(k_B * T / sim.masa_particula)
    v_range = np.linspace(-3*sigma, 3*sigma, 1000)
    gauss = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-v_range**2/(2*sigma**2))
    
    ax1.plot(v_range, gauss, 'k--', linewidth=2, label='Gaussiana teórica')
    ax1.set_xlabel('Velocidad (m/s)', fontsize=12)
    ax1.set_ylabel('Densidad de Probabilidad', fontsize=12)
    ax1.set_title('Distribución de Componentes de Velocidad', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Distribución de magnitudes de velocidad (Maxwell 2D)
    ax2.hist(velocidades_magnitud, bins=20, alpha=0.7, density=True, color='green', label='Simulación')
    
    # Distribución de Maxwell 2D: f(v) = (m/(k_B*T)) * v * exp(-m*v²/(2*k_B*T))
    v_mag_range = np.linspace(0, max(velocidades_magnitud)*1.2, 1000)
    maxwell_2d = (sim.masa_particula/(k_B*T)) * v_mag_range * np.exp(-sim.masa_particula*v_mag_range**2/(2*k_B*T))
    
    ax2.plot(v_mag_range, maxwell_2d, 'r--', linewidth=2, label='Maxwell 2D teórica')
    ax2.set_xlabel('Magnitud de Velocidad (m/s)', fontsize=12)
    ax2.set_ylabel('Densidad de Probabilidad', fontsize=12)
    ax2.set_title('Distribución de Maxwell (2D)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        print(f"Figura guardada como {archivo}")
    
    plt.show()


def animar_particulas(sim, n_frames=200, intervalo=50, guardar=False, archivo='animacion.gif'):
    """
    Crea una animación de las partículas en movimiento.
    
    Args:
        sim: objeto SimulacionGasIdeal
        n_frames: número de frames
        intervalo: milisegundos entre frames
        guardar: si guardar la animación
        archivo: nombre del archivo
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Configurar ejes
    ax.set_xlim(sim.limites_x[0], sim.limites_x[1])
    ax.set_ylim(sim.limites_y[0], sim.limites_y[1])
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('Simulación de Gas Ideal en 2D', fontsize=14, fontweight='bold')
    
    # Dibujar caja
    caja = Rectangle((sim.limites_x[0], sim.limites_y[0]), 
                      sim.lado_caja, sim.lado_caja,
                      fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(caja)
    
    # Crear círculos para cada partícula
    circulos = []
    for p in sim.particulas:
        circulo = Circle(p.posicion, p.radio, color='blue', alpha=0.7)
        ax.add_patch(circulo)
        circulos.append(circulo)
    
    # Texto para información
    texto_info = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    dt = 1e-13  # Paso de tiempo
    
    def actualizar(frame):
        # Avanzar simulación
        for _ in range(10):  # Varios pasos por frame para acelerar
            sim.paso_temporal(dt)
        
        # Actualizar posiciones de círculos
        for circulo, particula in zip(circulos, sim.particulas):
            circulo.center = particula.posicion
        
        # Actualizar información
        E = sim.energia_total()
        T = sim.temperatura()
        v_rms = sim.velocidad_cuadratica_media()
        
        info = f'Tiempo: {sim.t_actual*1e9:.3f} ns\n'
        info += f'Energía: {E:.3e} J\n'
        info += f'Temperatura: {T:.1f} K\n'
        info += f'$v_{{rms}}$: {v_rms:.1f} m/s'
        texto_info.set_text(info)
        
        return circulos + [texto_info]
    
    anim = animation.FuncAnimation(fig, actualizar, frames=n_frames,
                                   interval=intervalo, blit=True, repeat=True)
    
    if guardar:
        print(f"Guardando animación como {archivo}...")
        anim.save(archivo, writer='pillow', fps=20, dpi=100)
        print(f"Animación guardada como {archivo}")
    
    plt.show()


def reporte_completo(sim, guardar_figuras=False):
    """
    Genera un reporte completo con todas las gráficas y estadísticas.
    
    Args:
        sim: objeto SimulacionGasIdeal con historial
        guardar_figuras: si guardar las figuras
    """
    print("=" * 70)
    print("REPORTE DE SIMULACIÓN DE GAS IDEAL")
    print("=" * 70)
    print()
    
    # Estadísticas
    stats = sim.obtener_estadisticas()
    
    print("PARÁMETROS DE SIMULACIÓN:")
    print(f"  Número de partículas: {sim.n_particulas}")
    print(f"  Lado de la caja: {sim.lado_caja*1e6:.2f} μm")
    print(f"  Masa de partícula: {sim.masa_particula:.3e} kg")
    print(f"  Tiempo simulado: {stats['tiempo_total']*1e9:.3f} ns")
    print()
    
    print("RESULTADOS:")
    print(f"  Energía media: {stats['energia_media']:.6e} J")
    print(f"  Desviación energía: {stats['energia_std']:.6e} J")
    print(f"  Variación energía: {stats['energia_variacion']:.4f}%")
    print()
    print(f"  Temperatura media: {stats['temperatura_media']:.2f} K")
    print(f"  Desviación temperatura: {stats['temperatura_std']:.2f} K")
    print()
    print(f"  v_rms simulada: {stats['v_rms_media']:.2f} m/s")
    print(f"  v_rms teórica: {stats['v_rms_teorica']:.2f} m/s")
    print(f"  Error relativo: {abs(stats['v_rms_media']-stats['v_rms_teorica'])/stats['v_rms_teorica']*100:.2f}%")
    print()
    
    # Verificaciones
    conserva, variacion = sim.verificar_conservacion_energia()
    print("VERIFICACIÓN DE FÍSICA:")
    print(f"  Conservación de energía: {'✓ SÍ' if conserva else '✗ NO'}")
    print(f"  Variación de energía: {variacion:.4f}%")
    
    cumple, error = sim.verificar_relacion_temperatura_velocidad()
    print(f"  Relación T-v correcta: {'✓ SÍ' if cumple else '✗ NO'}")
    print(f"  Error en relación T-v: {error*100:.2f}%")
    print()
    print("=" * 70)
    
    # Generar gráficas
    print("\nGenerando gráficas...")
    graficar_conservacion_energia(sim, guardar_figuras, 'conservacion_energia.png')
    graficar_temperatura_velocidad(sim, guardar_figuras, 'temperatura_velocidad.png')
    graficar_distribucion_velocidades(sim, guardar_figuras, 'distribucion_velocidades.png')


if __name__ == '__main__':
    print("Creando y ejecutando simulación...")
    
    # Crear simulación
    sim = SimulacionGasIdeal(n_particulas=50, lado_caja=1e-6, temperatura_inicial=300)
    
    # Ejecutar simulación
    sim.ejecutar(t_total=1e-10, dt=1e-13, guardar_cada=10)
    
    # Generar reporte
    reporte_completo(sim, guardar_figuras=True)
    
    # Opcional: crear animación (comentar si no se desea)
    # print("\nCreando animación...")
    # sim_anim = SimulacionGasIdeal(n_particulas=30, lado_caja=1e-6, temperatura_inicial=300)
    # animar_particulas(sim_anim, n_frames=100, guardar=True)