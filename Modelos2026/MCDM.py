# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:24:35 2026

@author: hegue
"""
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys

class AHP_Pesos:
    def __init__(self):
        # Índices de Consistencia Aleatoria (RI) según Saaty
        self.RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

    def obtener_pesos(self, matriz_comparacion, nombres_criterios):
        """
        Calcula exclusivamente los pesos y valida la consistencia.
        """
        matriz = np.array(matriz_comparacion)
        n = matriz.shape[0]
        
        # 1. Cálculo del Vector de Prioridad (Pesos)
        suma_columnas = matriz.sum(axis=0)
        matriz_norm = matriz / suma_columnas
        pesos = matriz_norm.mean(axis=1)
        
        # 2. Análisis de Consistencia (RC)
        valor_v = np.dot(matriz, pesos)
        lambda_max = np.mean(valor_v / pesos)
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0
        ri = self.RI.get(n, 1.49)
        cr = ci / ri if ri > 0 else 0
        
        # --- REPORTE EN CONSOLA ---
        print("\n" + "="*45)
        print("   ANÁLISIS DE IMPORTANCIA RELATIVA (AHP)")
        print("="*45)
        print(f"Consistencia (RC): {cr:.4f}")
        
        if cr < 0.10:
            print("Estado: ✅ Matriz Consistente")
        else:
            print("Estado: ⚠️ Revisar Matriz (Inconsistente)")
        print("-" * 45)
        
        # Crear un DataFrame para visualizar los pesos claramente
        df_pesos = pd.DataFrame({
            'Criterio': nombres_criterios,
            'Peso_Decimal': pesos,
            'Peso_Porcentual': [f"{p*100:.2f}%" for p in pesos]
        })
        
        return df_pesos

# ==========================================
# EJECUCIÓN: SOLO CÁLCULO DE PESOS
# ==========================================
if __name__ == "__main__":
    # 1. Definir los nombres de tus criterios
    mis_criterios = ['Rentabilidad', 'Riesgo', 'Costo']

    # 2. Matriz de Saaty (Tus comparaciones)
    matriz_saaty = [
        [1,   3,   5], 
        [1/3, 1,   2], 
        [1/5, 1/2, 1]
    ]

    # 3. Inicializar y procesar
    ahp = AHP_Pesos()
    tabla_pesos = ahp.obtener_pesos(matriz_saaty, mis_criterios)

    # 4. Mostrar solo los pesos resultantes
    print(tabla_pesos.to_string(index=False))
    print("="*45)