import unittest
import numpy as np
from particula import Particula
from simulacion import SimulacionGasIdeal

class TestParticula(unittest.TestCase):
    """Pruebas unitarias para la clase Partícula"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        self.masa = 1.0
        self.posicion = [0.5, 0.5]
        self.velocidad = [10.0, 5.0]
        self.particula = Particula(self.masa, self.posicion, self.velocidad)
    
    def test_inicializacion(self):
        """Prueba la inicialización de la partícula"""
        self.assertEqual(self.particula.masa, self.masa)
        np.testing.assert_array_equal(self.particula.posicion, np.array(self.posicion))
        np.testing.assert_array_equal(self.particula.velocidad, np.array(self.velocidad))
    
    def test_energia_cinetica(self):
        """Prueba el cálculo de energía cinética"""
        # E = 1/2 * m * v²
        v_magnitud = np.sqrt(10**2 + 5**2)
        energia_esperada = 0.5 * self.masa * v_magnitud**2
        self.assertAlmostEqual(self.particula.energia_cinetica(), energia_esperada)
    
    def test_momento(self):
        """Prueba el cálculo del momento lineal"""
        momento_esperado = self.masa * np.array(self.velocidad)
        np.testing.assert_array_almost_equal(self.particula.momento(), momento_esperado)
    
    def test_actualizacion_posicion(self):
        """Prueba la actualización de posición"""
        dt = 0.1
        posicion_inicial = self.particula.posicion.copy()
        self.particula.actualizar_posicion(dt)
        
        posicion_esperada = posicion_inicial + np.array(self.velocidad) * dt
        np.testing.assert_array_almost_equal(self.particula.posicion, posicion_esperada)
    
    def test_colision_pared_izquierda(self):
        """Prueba colisión con pared izquierda"""
        p = Particula(1.0, [0.005, 0.5], [-10.0, 0.0], radio=0.01)
        limites_x = (0, 1)
        limites_y = (0, 1)
        
        colision = p.colision_pared(limites_x, limites_y)
        
        self.assertTrue(colision)
        self.assertGreater(p.velocidad[0], 0)  # Velocidad debe invertirse (positiva)
    
    def test_colision_pared_derecha(self):
        """Prueba colisión con pared derecha"""
        p = Particula(1.0, [0.995, 0.5], [10.0, 0.0], radio=0.01)
        limites_x = (0, 1)
        limites_y = (0, 1)
        
        colision = p.colision_pared(limites_x, limites_y)
        
        self.assertTrue(colision)
        self.assertLess(p.velocidad[0], 0)  # Velocidad debe invertirse (negativa)
    
    def test_colision_particulas(self):
        """Prueba colisión entre dos partículas"""
        # Configuración: p1 a la derecha yendo izquierda, p2 a la izquierda yendo derecha
        # Ambas se acercan hacia el centro y colisionan
        p1 = Particula(1.0, [0.53, 0.5], [-10.0, 0.0], radio=0.02)
        p2 = Particula(1.0, [0.50, 0.5], [10.0, 0.0], radio=0.02)
        
        # Verificar que están superpuestas (distancia < suma de radios)
        distancia = np.linalg.norm(p1.posicion - p2.posicion)
        suma_radios = p1.radio + p2.radio
        print(f"Distancia: {distancia:.4f}, Suma radios: {suma_radios:.4f}")
        self.assertLess(distancia, suma_radios, "Las partículas deben estar superpuestas")
        
        # Guardar estado antes de colisión
        momento_total_antes = p1.momento() + p2.momento()
        energia_total_antes = p1.energia_cinetica() + p2.energia_cinetica()
        
        # Ejecutar colisión
        colision = p1.colision_particula(p2)
        
        # Verificar que hubo colisión
        self.assertTrue(colision, f"No hubo colisión. Distancia: {distancia:.4f}, Radios: {suma_radios:.4f}")
        
        # Verificar conservación de momento
        momento_total_despues = p1.momento() + p2.momento()
        np.testing.assert_array_almost_equal(momento_total_antes, momento_total_despues, decimal=5)
        
        # Verificar conservación de energía
        energia_total_despues = p1.energia_cinetica() + p2.energia_cinetica()
        self.assertAlmostEqual(energia_total_antes, energia_total_despues, places=5)


class TestSimulacion(unittest.TestCase):
    """Pruebas unitarias para la clase SimulacionGasIdeal"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        np.random.seed(42)  # Para reproducibilidad
        self.sim = SimulacionGasIdeal(n_particulas=10, lado_caja=1e-6, temperatura_inicial=300)
    
    def test_inicializacion(self):
        """Prueba la inicialización de la simulación"""
        self.assertEqual(len(self.sim.particulas), 10)
        self.assertEqual(self.sim.lado_caja, 1e-6)
    
    def test_momento_total_cero(self):
        """Verifica que el momento total inicial sea aproximadamente cero"""
        momento_total = sum(p.momento() for p in self.sim.particulas)
        self.assertAlmostEqual(np.linalg.norm(momento_total), 0, places=10)
    
    def test_temperatura_inicial(self):
        """Verifica que la temperatura inicial sea cercana a la esperada"""
        T = self.sim.temperatura()
        # Permitir un margen debido a fluctuaciones estadísticas
        self.assertAlmostEqual(T, 300, delta=50)
    
    def test_conservacion_energia(self):
        """Prueba la conservación de energía durante la simulación"""
        energia_inicial = self.sim.energia_total()
        
        # Ejecutar simulación corta
        dt = 1e-12
        for _ in range(100):
            self.sim.paso_temporal(dt)
        
        energia_final = self.sim.energia_total()
        
        # La energía debe conservarse con alta precisión
        diferencia_relativa = abs(energia_final - energia_inicial) / energia_inicial
        self.assertLess(diferencia_relativa, 1e-3)
    
    def test_particulas_dentro_caja(self):
        """Verifica que las partículas permanezcan dentro de la caja"""
        dt = 1e-12
        for _ in range(100):
            self.sim.paso_temporal(dt)
        
        for p in self.sim.particulas:
            self.assertGreaterEqual(p.posicion[0], self.sim.limites_x[0])
            self.assertLessEqual(p.posicion[0], self.sim.limites_x[1])
            self.assertGreaterEqual(p.posicion[1], self.sim.limites_y[0])
            self.assertLessEqual(p.posicion[1], self.sim.limites_y[1])
    
    def test_relacion_temperatura_velocidad(self):
        """Prueba la relación entre temperatura y velocidad cuadrática media"""
        # Ejecutar simulación para alcanzar equilibrio
        self.sim.ejecutar(t_total=1e-10, dt=1e-13, guardar_cada=10)
        
        T = np.mean(self.sim.historial_temperatura)
        v_rms = np.mean(self.sim.historial_velocidades)
        
        # v_rms teórica = sqrt(2*k_B*T/m) para 2D
        k_B = 1.380649e-23
        v_rms_teorica = np.sqrt(2 * k_B * T / self.sim.masa_particula)
        
        error_relativo = abs(v_rms - v_rms_teorica) / v_rms_teorica
        print(f"Error relativo T-v: {error_relativo*100:.2f}%")
        self.assertLess(error_relativo, 0.20)  # Error menor al 20% (tolerante para pocas partículas)
    
    def test_verificacion_conservacion_energia(self):
        """Prueba el método de verificación de conservación de energía"""
        self.sim.ejecutar(t_total=5e-11, dt=1e-13, guardar_cada=10)
        
        conserva, variacion = self.sim.verificar_conservacion_energia(tolerancia=0.02)
        
        self.assertTrue(conserva)
        self.assertLess(variacion, 2.0)  # Variación menor al 2%
    
    def test_estadisticas(self):
        """Prueba el cálculo de estadísticas"""
        self.sim.ejecutar(t_total=2e-11, dt=1e-13, guardar_cada=10)
        
        stats = self.sim.obtener_estadisticas()
        
        self.assertIn('energia_media', stats)
        self.assertIn('temperatura_media', stats)
        self.assertIn('v_rms_media', stats)
        self.assertGreater(stats['energia_media'], 0)
        self.assertGreater(stats['temperatura_media'], 0)


class TestFisicaGasIdeal(unittest.TestCase):
    """Pruebas de validez física del modelo"""
    
    def test_distribucion_velocidades_maxwelliana(self):
        """Verifica que la distribución de velocidades sea aproximadamente Maxwelliana"""
        np.random.seed(42)
        sim = SimulacionGasIdeal(n_particulas=100, lado_caja=1e-6, temperatura_inicial=300)
        
        # Recolectar velocidades
        velocidades_x = [p.velocidad[0] for p in sim.particulas]
        
        # La media debe ser cercana a cero
        media_vx = np.mean(velocidades_x)
        self.assertAlmostEqual(media_vx, 0, delta=50)
        
        # La desviación estándar debe seguir la teoría
        # sigma = sqrt(k_B*T/m)
        k_B = 1.380649e-23
        sigma_teorica = np.sqrt(k_B * 300 / sim.masa_particula)
        sigma_real = np.std(velocidades_x)
        
        error_relativo = abs(sigma_real - sigma_teorica) / sigma_teorica
        self.assertLess(error_relativo, 0.3)  # Error menor al 30%
    
    def test_presion_cualitativa(self):
        """Verifica cualitativamente que hay colisiones con paredes (presión)"""
        np.random.seed(42)
        # Crear simulación con condiciones que garanticen colisiones
        sim = SimulacionGasIdeal(n_particulas=50, lado_caja=1e-6, temperatura_inicial=1000)
        
        # Contar cambios de dirección (rebotes)
        rebotes_x = 0
        rebotes_y = 0
        
        # Guardar velocidades iniciales
        vel_ant = [p.velocidad.copy() for p in sim.particulas]
        
        for paso in range(5000):
            sim.paso_temporal(5e-14)
            
            # Contar cambios de signo en velocidades (indica rebote)
            for i, p in enumerate(sim.particulas):
                if vel_ant[i][0] * p.velocidad[0] < 0:  # Cambió de signo en x
                    rebotes_x += 1
                if vel_ant[i][1] * p.velocidad[1] < 0:  # Cambió de signo en y
                    rebotes_y += 1
                vel_ant[i] = p.velocidad.copy()
        
        total_rebotes = rebotes_x + rebotes_y
        print(f"Rebotes detectados: {total_rebotes} (x: {rebotes_x}, y: {rebotes_y})")
        
        # Con 50 partículas y 5000 pasos debe haber muchos rebotes
        self.assertGreater(total_rebotes, 100)


def suite():
    """Crea una suite con todas las pruebas"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestParticula))
    suite.addTest(unittest.makeSuite(TestSimulacion))
    suite.addTest(unittest.makeSuite(TestFisicaGasIdeal))
    return suite


if __name__ == '__main__':
    print("=" * 70)
    print("EJECUTANDO PRUEBAS UNITARIAS DEL SIMULADOR DE GAS IDEAL")
    print("=" * 70)
    print()
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    
    print()
    print("=" * 70)
    print("RESUMEN DE PRUEBAS")
    print("=" * 70)
    print(f"Pruebas ejecutadas: {result.testsRun}")
    print(f"Exitosas: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallidas: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")
    print("=" * 70)