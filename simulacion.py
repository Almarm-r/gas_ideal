import numpy as np
from particula import Particula

class SimulacionGasIdeal:
    """
    Clase para simular un gas ideal en una caja cuadrada.
    """
    
    def __init__(self, n_particulas, lado_caja, masa_particula=1e-26, temperatura_inicial=300):
        """
        Inicializa la simulación.
        
        Args:
            n_particulas: número de partículas
            lado_caja: lado de la caja cuadrada (m)
            masa_particula: masa de cada partícula (kg)
            temperatura_inicial: temperatura inicial del sistema (K)
        """
        self.n_particulas = n_particulas
        self.lado_caja = lado_caja
        self.masa_particula = masa_particula
        self.limites_x = (0, lado_caja)
        self.limites_y = (0, lado_caja)
        
        # Constante de Boltzmann
        self.k_B = 1.380649e-23  # J/K
        
        # Crear partículas
        self.particulas = self._inicializar_particulas(temperatura_inicial)
        
        # Historial para análisis
        self.historial_energia = []
        self.historial_temperatura = []
        self.historial_velocidades = []
        self.tiempo = []
        self.t_actual = 0
    
    def _inicializar_particulas(self, temperatura):
        """
        Inicializa las partículas con distribución de Maxwell-Boltzmann.
        
        Args:
            temperatura: temperatura inicial (K)
        
        Returns:
            list: lista de objetos Partícula
        """
        particulas = []
        
        # Velocidad cuadrática media según teoría cinética para 2D
        # v_rms = sqrt(2*k_B*T/m)
        v_rms = np.sqrt(2 * self.k_B * temperatura / self.masa_particula)
        
        # Distribución espacial uniforme (evitando superposición)
        n_filas = int(np.ceil(np.sqrt(self.n_particulas)))
        espaciado = self.lado_caja / (n_filas + 1)
        radio = espaciado * 0.3  # Radio de partícula
        
        for i in range(self.n_particulas):
            fila = i // n_filas
            col = i % n_filas
            
            # Posición inicial con pequeña aleatoriedad
            x = (col + 1) * espaciado + np.random.uniform(-espaciado*0.1, espaciado*0.1)
            y = (fila + 1) * espaciado + np.random.uniform(-espaciado*0.1, espaciado*0.1)
            posicion = [x, y]
            
            # Velocidad según distribución de Maxwell-Boltzmann
            # Cada componente sigue una distribución normal con sigma = sqrt(k_B*T/m)
            sigma = np.sqrt(self.k_B * temperatura / self.masa_particula)
            vx = np.random.normal(0, sigma)
            vy = np.random.normal(0, sigma)
            velocidad = [vx, vy]
            
            particulas.append(Particula(self.masa_particula, posicion, velocidad, radio))
        
        # Ajustar para momento total nulo (centro de masa en reposo)
        momento_total = sum(p.momento() for p in particulas)
        velocidad_cm = momento_total / (self.n_particulas * self.masa_particula)
        for p in particulas:
            p.velocidad -= velocidad_cm
        
        return particulas
    
    def energia_total(self):
        """
        Calcula la energía cinética total del sistema.
        
        Returns:
            float: Energía total en Joules
        """
        return sum(p.energia_cinetica() for p in self.particulas)
    
    def temperatura(self):
        """
        Calcula la temperatura instantánea del sistema.
        Usando el teorema de equipartición: <E_cin> = (N_gl/2) * k_B * T
        Para N partículas en 2D: N_gl = 2N (dos grados de libertad por partícula)
        Por tanto: E_total = N * k_B * T
        
        Returns:
            float: Temperatura en Kelvin
        """
        energia_promedio = self.energia_total() / self.n_particulas
        # En 2D: (1/2)m<v²> = k_B*T
        # Por tanto: T = m<v²> / (2*k_B)
        v2_promedio = sum(np.sum(p.velocidad**2) for p in self.particulas) / self.n_particulas
        return self.masa_particula * v2_promedio / (2 * self.k_B)
    
    def velocidad_cuadratica_media(self):
        """
        Calcula la velocidad cuadrática media del sistema.
        v_rms = sqrt(<v²>)
        
        Returns:
            float: Velocidad cuadrática media en m/s
        """
        v2_promedio = sum(np.sum(p.velocidad**2) for p in self.particulas) / self.n_particulas
        return np.sqrt(v2_promedio)
    
    def paso_temporal(self, dt):
        """
        Avanza la simulación un paso de tiempo.
        
        Args:
            dt: paso de tiempo (s)
        """
        # Actualizar posiciones
        for p in self.particulas:
            p.actualizar_posicion(dt)
        
        # Manejar colisiones con paredes
        for p in self.particulas:
            p.colision_pared(self.limites_x, self.limites_y)
        
        # Manejar colisiones entre partículas
        for i in range(len(self.particulas)):
            for j in range(i+1, len(self.particulas)):
                self.particulas[i].colision_particula(self.particulas[j])
        
        self.t_actual += dt
    
    def ejecutar(self, t_total, dt, guardar_cada=10):
        """
        Ejecuta la simulación durante un tiempo total.
        
        Args:
            t_total: tiempo total de simulación (s)
            dt: paso de tiempo (s)
            guardar_cada: guardar datos cada N pasos
        """
        n_pasos = int(t_total / dt)
        
        for paso in range(n_pasos):
            self.paso_temporal(dt)
            
            # Guardar datos para análisis
            if paso % guardar_cada == 0:
                self.tiempo.append(self.t_actual)
                self.historial_energia.append(self.energia_total())
                self.historial_temperatura.append(self.temperatura())
                self.historial_velocidades.append(self.velocidad_cuadratica_media())
    
    def obtener_estadisticas(self):
        """
        Calcula estadísticas del sistema.
        
        Returns:
            dict: diccionario con estadísticas
        """
        energia = np.array(self.historial_energia)
        temperatura = np.array(self.historial_temperatura)
        
        return {
            'energia_media': np.mean(energia),
            'energia_std': np.std(energia),
            'energia_variacion': (np.max(energia) - np.min(energia)) / np.mean(energia) * 100,
            'temperatura_media': np.mean(temperatura),
            'temperatura_std': np.std(temperatura),
            'v_rms_media': np.mean(self.historial_velocidades),
            'v_rms_teorica': np.sqrt(2 * self.k_B * np.mean(temperatura) / self.masa_particula),
            'tiempo_total': self.t_actual
        }
    
    def verificar_conservacion_energia(self, tolerancia=1e-2):
        """
        Verifica si la energía se conserva dentro de una tolerancia.
        
        Args:
            tolerancia: tolerancia relativa para la conservación (fracción)
        
        Returns:
            tuple: (bool conserva, float variacion_porcentual)
        """
        energia = np.array(self.historial_energia)
        variacion = (np.max(energia) - np.min(energia)) / np.mean(energia)
        return variacion < tolerancia, variacion * 100
    
    def verificar_relacion_temperatura_velocidad(self, tolerancia=5e-2):
        """
        Verifica la relación teórica entre temperatura y velocidad.
        Teoría en 2D: v_rms = sqrt(2*k_B*T/m)
        
        Args:
            tolerancia: tolerancia relativa para la verificación
        
        Returns:
            tuple: (bool cumple, float error_relativo)
        """
        T = np.mean(self.historial_temperatura)
        v_rms_simulada = np.mean(self.historial_velocidades)
        v_rms_teorica = np.sqrt(2 * self.k_B * T / self.masa_particula)
        
        error_relativo = abs(v_rms_simulada - v_rms_teorica) / v_rms_teorica
        return error_relativo < tolerancia, error_relativo


def crear_simulacion_ejemplo(n_particulas=50, lado=1e-6, T=300):
    """
    Crea una simulación de ejemplo.
    
    Args:
        n_particulas: número de partículas
        lado: lado de la caja (m)
        T: temperatura inicial (K)
    
    Returns:
        SimulacionGasIdeal: objeto de simulación
    """
    return SimulacionGasIdeal(n_particulas, lado, temperatura_inicial=T)