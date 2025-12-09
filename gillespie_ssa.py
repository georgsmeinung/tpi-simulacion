import numpy as np
import matplotlib.pyplot as plt

# Funcion de Simulacion
def gillespie_ssa(k1, k2, t_max):
    # Inicialización
    t = 0.0
    # Condición inicial (puedes cambiarla)
    mRNA = 0
    
    # Listas para guardar el historial (para graficar)
    time_points = [t]
    mRNA_counts = [mRNA]
    
    while t < t_max:
        # 1. Calcular las propensiones (propensities)
        # a1: Probabilidad de producción (constante)
        a1 = k1
        # a2: Prob. degradación (proporcional a moléculas)
        a2 = k2 * mRNA
        
        a_sum = a1 + a2
        
        # Si a_sum es 0, el sistema para (sin reacciones)
        if a_sum == 0:
            break
            
        # 2. Determinar tiempo hasta próxima reacción (tau)
        # Se extrae de una distribución exponencial
        r1 = np.random.rand()
        tau = (1.0 / a_sum) * np.log(1.0 / r1)
        
        # 3. Determinar qué reacción ocurre
        r2 = np.random.rand()
        
        if r2 < (a1 / a_sum):
            # Ocurre Reacción 1: Producción
            mRNA += 1
        else:
            # Ocurre Reacción 2: Degradación
            mRNA -= 1
            
        # 4. Actualizar tiempo y guardar estado
        t += tau
        time_points.append(t)
        mRNA_counts.append(mRNA)
        
    return time_points, mRNA_counts

# Función para Generar el Informe
def generar_informe_texto(tiempos, conteos, k1, k2):
    # Convertir a numpy array para cálculos estadísticos
    arr_conteos = np.array(conteos)
    
    # Cálculos estadísticos sobre la simulación
    media_sim = np.mean(arr_conteos)
    desv_std = np.std(arr_conteos)
    var_sim = np.var(arr_conteos)
    min_val = np.min(arr_conteos)
    max_val = np.max(arr_conteos)
    
    total_ev = len(tiempos) - 1
    t_final = tiempos[-1]
    
    # Valores Teóricos
    media_teorica = k1 / k2
    # En Poisson ideal, varianza es igual a la media
    var_teorica = media_teorica 
    
    # Cálculo del error relativo (dividido en pasos)
    diff = abs(media_sim - media_teorica)
    error_rel = (diff / media_teorica) * 100

    # Definimos separadores para usar dentro del f-string
    sep = "=" * 50
    sub = "-" * 50

    # Construcción del reporte usando concatenación implícita
    # para respetar el ancho de 70 caracteres en el editor
    informe = (
        f"\n{sep}\n"
        f"SIMULACIÓN ESTOCÁSTICA (GILLESPIE SSA)\n"
        f"{sep}\n\n"
        f"1. PARÁMETROS DEL SISTEMA\n"
        f"{sub}\n"
        f"Reacción 1 (Producción) k1 : {k1}\n"
        f"Reacción 2 (Degradación) k2: {k2}\n"
        f"Tiempo total simulado      : {t_final:.2f} u.t.\n"
        f"Total de eventos ocurridos : {total_ev}\n\n"
        f"2. ANÁLISIS ESTADÍSTICO DE LA TRAYECTORIA\n"
        f"{sub}\n"
        f"Media observada            : {media_sim:.4f} mols\n"
        f"Desviación estándar        : {desv_std:.4f}\n"
        f"Varianza observada         : {var_sim:.4f}\n"
        f"Rango de fluctuación       : [{min_val}] - [{max_val}]\n\n"
        f"3. VALIDACIÓN CON MODELO TEÓRICO\n"
        f"{sub}\n"
        f"Estado Estacionario Esp.   : {media_teorica:.4f} mols\n"
        f"Varianza Esperada (Poisson): {var_teorica:.4f}\n"
        f"Error Relativo (Media)     : {error_rel:.2f}%\n"
    )
    
    return informe