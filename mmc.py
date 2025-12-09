import random
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import Image, display

class SimulacionMMC_Log:
    def __init__(self, tiempo_max, tasa_llegada,
                 tasa_servicio, n_servidores):
        # Ajustamos los parámetros iniciales
        self.tiempo_max = tiempo_max
        self.tasa_llegada = tasa_llegada
        self.tasa_servicio = tasa_servicio
        self.n_servidores = n_servidores
        
        # Estado del sistema
        self.reloj = 0.0
        self.num_en_cola = 0
        self.servidores_ocupados = 0
        self.lef = []  # Lista de Eventos Futuros
        
        # Estadísticas y Acumuladores
        self.total_llegadas = 0
        self.total_atendidos = 0
        self.area_cola = 0.0      # Para calcular Lq
        self.area_ocupados = 0.0  # Para calcular utilización
        self.tiempo_ultimo_evento = 0.0
        
        # Historial para Animación
        self.historia = [(0.0, 0, 0)]

    def actualizar_estadisticas(self, tiempo_actual):
        """Calcula áreas bajo la curva antes de cambiar estado."""
        delta_t = tiempo_actual - self.tiempo_ultimo_evento
        self.area_cola += self.num_en_cola * delta_t
        self.area_ocupados += self.servidores_ocupados * delta_t
        self.tiempo_ultimo_evento = tiempo_actual

    def correr(self):
        # Programar primera llegada
        t_llegada = random.expovariate(self.tasa_llegada)
        heapq.heappush(self.lef, (t_llegada, 0))  # 0 = LLEGADA
        
        msg = (f"--- Iniciando Simulación M/M/{self.n_servidores} "
               f"por {self.tiempo_max} horas ---")
        print(msg)
        
        while self.reloj < self.tiempo_max and self.lef:
            tiempo_evento, tipo_evento = heapq.heappop(self.lef)
            
            if tiempo_evento > self.tiempo_max:
                # Actualizar hasta tiempo final exacto
                self.actualizar_estadisticas(self.tiempo_max)
                self.reloj = self.tiempo_max
                break
            
            # 1. Actualizar estadísticas
            self.actualizar_estadisticas(tiempo_evento)
            
            # 2. Avanzar reloj
            self.reloj = tiempo_evento
            
            # 3. Procesar evento
            if tipo_evento == 0:  # LLEGADA
                self.procesar_llegada()
            else:  # SALIDA
                self.procesar_salida()
            
            # 4. Guardar foto para la animación
            self.historia.append((
                self.reloj,
                self.num_en_cola,
                self.servidores_ocupados
            ))

        # AL FINALIZAR EL BUCLE: MOSTRAR RESULTADOS
        self.reporte_final()

    def procesar_llegada(self):
        self.total_llegadas += 1
        prox = self.reloj + random.expovariate(self.tasa_llegada)
        heapq.heappush(self.lef, (prox, 0))
        
        if self.servidores_ocupados < self.n_servidores:
            self.servidores_ocupados += 1
            t_salida = self.reloj + random.expovariate(
                self.tasa_servicio
            )
            heapq.heappush(self.lef, (t_salida, 1))
        else:
            self.num_en_cola += 1

    def procesar_salida(self):
        self.total_atendidos += 1
        if self.num_en_cola > 0:
            self.num_en_cola -= 1
            t_salida = self.reloj + random.expovariate(
                self.tasa_servicio
            )
            heapq.heappush(self.lef, (t_salida, 1))
        else:
            self.servidores_ocupados -= 1

    def reporte_final(self):
        print("\n" + "="*40)
        print("      RESULTADOS DE LA SIMULACIÓN")
        print("="*40)
        print(f"Tiempo simulado: {self.reloj:.2f} horas")
        print(f"Total llegadas:  {self.total_llegadas}")
        print(f"Total atendidos: {self.total_atendidos}")
        print("-" * 40)
        
        # Cálculos de promedios
        lq = self.area_cola / self.reloj
        
        prom_ocupados = self.area_ocupados / self.reloj
        utilizacion = prom_ocupados / self.n_servidores
        
        # Wq usando fórmula de Little
        lambda_real = self.total_llegadas / self.reloj
        wq = lq / lambda_real if lambda_real > 0 else 0
        
        print(f"Longitud prom cola (Lq): {lq:.4f} clientes")
        print(f"Tiempo prom cola (Wq): {wq*60:.2f} minutos")
        print(f"Servidores ocupados (prom): {prom_ocupados:.2f}")
        print(f"Utilización (Rho): {utilizacion*100:.2f}%")
        print("="*40 + "\n")

    def graficar_historial(self):
        """Gráficos estáticos de evolución de cola y servidores."""
        # 1. Desempaquetar datos
        tiempos = [d[0] for d in self.historia]
        n_cola = [d[1] for d in self.historia]
        n_ocupados = [d[2] for d in self.historia]

        # 2. Configurar figura (2 subplots)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True
        )

        # --- Gráfico 1: Evolución de la Cola ---
        ax1.step(tiempos, n_cola, where='post',
                 color='tab:blue', linewidth=1.5)
        ax1.fill_between(tiempos, n_cola, step='post',
                         alpha=0.2, color='tab:blue')
        ax1.set_ylabel('Clientes en Cola')
        ax1.set_title(
            f'Evolución Cola (M/M/{self.n_servidores})'
        )
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- Gráfico 2: Uso de Servidores ---
        ax2.step(tiempos, n_ocupados, where='post',
                 color='tab:orange', linewidth=1.5)
        ax2.fill_between(tiempos, n_ocupados, step='post',
                         alpha=0.2, color='tab:orange')
        ax2.set_ylabel('Servidores Ocupados')
        ax2.set_xlabel('Tiempo de Simulación (horas)')
        ax2.set_title('Ocupación de Servidores')
        
        # Línea de capacidad máxima
        ax2.axhline(y=self.n_servidores, color='red',
                    linestyle=':', label='Capacidad Max')
        ax2.set_ylim(0, self.n_servidores + 0.5)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()