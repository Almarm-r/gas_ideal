import numpy as np

class Particula:
    """
    Clase que representa una partícula de gas ideal.
    
    Attributes:
        masa (float): Masa de la partícula en kg
        posicion (np.array): Vector posición [x, y] en metros
        velocidad (np.array): Vector velocidad [vx, vy] en m/s
        radio (float): Radio de la partícula para visualización
    """
    
    def __init__(self, masa, posicion, velocidad, radio=0.01):
        """
        Inicializa una partícula.
        
        Args:
            masa: masa de la partícula (kg)
            posicion: array [x, y] con la posición inicial (m)
            velocidad: array [vx, vy] con la velocidad inicial (m/s)
            radio: radio de la partícula (m)
        """
        self.masa = masa
        self.posicion = np.array(posicion, dtype=float)
        self.velocidad = np.array(velocidad, dtype=float)
        self.radio = radio
    
    def actualizar_posicion(self, dt):
        """
        Actualiza la posición de la partícula usando integración de Euler.
        
        Args:
            dt: paso de tiempo (s)
        """
        self.posicion += self.velocidad * dt
    
    def energia_cinetica(self):
        """
        Calcula la energía cinética de la partícula.
        
        Returns:
            float: Energía cinética en Joules
        """
        v_magnitud = np.linalg.norm(self.velocidad)
        return 0.5 * self.masa * v_magnitud**2
    
    def momento(self):
        """
        Calcula el momento lineal de la partícula.
        
        Returns:
            np.array: Vector momento [px, py] en kg·m/s
        """
        return self.masa * self.velocidad
    
    def velocidad_cuadratica_media(self):
        """
        Calcula la velocidad cuadrática media de esta partícula.
        
        Returns:
            float: Velocidad cuadrática media en m/s
        """
        return np.sqrt(np.sum(self.velocidad**2))
    
    def colision_pared(self, limite_x, limite_y):
        """
        Detecta y maneja colisiones elásticas con las paredes de la caja.
        
        Args:
            limite_x: tupla (x_min, x_max) con los límites en x
            limite_y: tupla (y_min, y_max) con los límites en y
        
        Returns:
            bool: True si hubo colisión, False si no
        """
        colision = False
        
        # Colisión con paredes verticales (izquierda/derecha)
        if self.posicion[0] <= limite_x[0] + self.radio:
            self.posicion[0] = limite_x[0] + self.radio
            self.velocidad[0] = abs(self.velocidad[0])  # Rebote hacia la derecha
            colision = True
        elif self.posicion[0] >= limite_x[1] - self.radio:
            self.posicion[0] = limite_x[1] - self.radio
            self.velocidad[0] = -abs(self.velocidad[0])  # Rebote hacia la izquierda
            colision = True
        
        # Colisión con paredes horizontales (inferior/superior)
        if self.posicion[1] <= limite_y[0] + self.radio:
            self.posicion[1] = limite_y[0] + self.radio
            self.velocidad[1] = abs(self.velocidad[1])  # Rebote hacia arriba
            colision = True
        elif self.posicion[1] >= limite_y[1] - self.radio:
            self.posicion[1] = limite_y[1] - self.radio
            self.velocidad[1] = -abs(self.velocidad[1])  # Rebote hacia abajo
            colision = True
        
        return colision
    
    def colision_particula(self, otra):
        """
        Detecta y maneja colisión elástica con otra partícula.
        
        Args:
            otra: otra instancia de Partícula
        
        Returns:
            bool: True si hubo colisión, False si no
        """
        # Vector de separación
        delta_pos = self.posicion - otra.posicion
        distancia = np.linalg.norm(delta_pos)
        
        # Verificar si hay colisión (distancia menor que suma de radios)
        if distancia < (self.radio + otra.radio) and distancia > 0:
            # Vector normal unitario
            n = delta_pos / distancia
            
            # Velocidades relativas
            delta_vel = self.velocidad - otra.velocidad
            
            # Velocidad relativa en dirección normal
            v_rel_n = np.dot(delta_vel, n)
            
            # Solo colisionan si se están acercando (v_rel_n < 0 significa acercamiento)
            # Si v_rel_n > 0, las partículas se alejan, no hay colisión
            if v_rel_n < 0:
                # Colisión elástica 2D (conservación de momento y energía)
                # Impulso intercambiado (usar valor absoluto ya que v_rel_n es negativo)
                impulso = (2 * abs(v_rel_n)) / (1/self.masa + 1/otra.masa)
                
                # Actualizar velocidades
                self.velocidad -= (impulso / self.masa) * n
                otra.velocidad += (impulso / otra.masa) * n
                
                # Separar partículas para evitar overlap
                overlap = (self.radio + otra.radio) - distancia
                separacion = n * (overlap / 2 + 0.001)
                self.posicion += separacion
                otra.posicion -= separacion
                
                return True
        
        return False
    
    def __repr__(self):
        return f"Partícula(masa={self.masa:.2e}, pos={self.posicion}, vel={self.velocidad})"