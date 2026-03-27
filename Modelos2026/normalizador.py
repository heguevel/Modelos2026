# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from tqdm import tqdm

# Configuración de estilo de gráficos
sns.set_theme(style="whitegrid")

class ModelosNormalizacion:
    def __init__(self):
        self.dir_originales = "Gráficos_Originales"
        self.dir_normalizados = "Gráficos_Normalizados"
        self._crear_directorios()

    def _crear_directorios(self):
        """Crea las carpetas base si no existen."""
        for carpeta in [self.dir_originales, self.dir_normalizados]:
            if not os.path.exists(carpeta):
                os.makedirs(carpeta)

    def _limpiar_directorios(self):
        """Borra archivos antiguos para evitar confusión de resultados."""
        for carpeta in [self.dir_originales, self.dir_normalizados]:
            if os.path.exists(carpeta):
                shutil.rmtree(carpeta)
            os.makedirs(carpeta)
        print(">>> Directorios de salida listos (limpios).")

    def _aplicar_reciproca(self, df, columnas):
        """Aplica 1/x para criterios minimizantes."""
        df_temp = df.copy()
        for col in columnas:
            if col in df_temp.columns:
                df_temp[col] = 1 / df_temp[col].replace(0, 1e-9)
        return df_temp

    def normalizacion_oecd_pro(self, df_procesado, columnas_num, n_intervalos=5):
        """Normalización basada en percentiles (OECD)."""
        df_oecd = df_procesado.copy()
        puntos_percentil = np.linspace(0, 100, n_intervalos + 1)
        puntajes = np.linspace(0, 100, n_intervalos + 1)
        for col in columnas_num:
            cortes = [np.percentile(df_procesado[col], p) for p in puntos_percentil]
            def asignar_categoria(x):
                for i in range(len(cortes) - 1):
                    if x <= cortes[i+1]: return puntajes[i]
                return puntajes[-1]
            df_oecd[col] = df_procesado[col].apply(asignar_categoria)
        return df_oecd

    def normalizacion_rim_pro(self, df_procesado, columnas_num, dict_metas=None):
        """
        Versión Corregida RIM: Garantiza 0.5 cuando el valor está a mitad de camino.
        """
        df_rim = df_procesado.copy()
        if dict_metas is None: return df_rim
        
        for col in columnas_num:
            if col in dict_metas:
                A, B, C, D = dict_metas[col]
                
                def calcular_f_x(x):
                    if C <= x <= D: return 1.0
                    elif x < C:
                        return 1.0 if A == C else 1 - (abs(x - C) / abs(A - C))
                    elif x > D:
                        return 1.0 if B == D else 1 - (abs(x - D) / abs(B - D))
                    return 0.0
                
                df_rim[col] = df_procesado[col].apply(calcular_f_x)
        return df_rim

    def calcular_normalizaciones(self, df, cols_num, minimo, metas_rim, n_intervalos_oecd):
        """Contenedor de lógica matemática pura."""
        df_p = self._aplicar_reciproca(df, minimo)
        
        metodos = {
            'Fracción Rango': (df_p[cols_num] - df_p[cols_num].min()) / (df_p[cols_num].max() - df_p[cols_num].min() + 1e-9),
            'Fracción Suma': df_p[cols_num] / (df_p[cols_num].sum() + 1e-9),
            'Fracción Máximo': df_p[cols_num] / (df_p[cols_num].max() + 1e-9),
            'Fracción Módulo': df_p[cols_num] / np.sqrt((df_p[cols_num]**2).sum() + 1e-9),
            'Z-Score': (df_p[cols_num] - df_p[cols_num].mean()) / (df_p[cols_num].std() + 1e-9),
            'Categórica': self.normalizacion_oecd_pro(df_p, cols_num, n_intervalos=n_intervalos_oecd),
            'Ideal de Referencia': self.normalizacion_rim_pro(df[cols_num], cols_num, metas_rim)
        }
        return metodos

    def generar_reporte_visual(self, df_original, dict_calculos, etiquetas, cols_num):
        """Orquesta la creación de gráficos Original vs Normalizado."""
        total_graficos = len(cols_num) + (len(dict_calculos) * len(cols_num))
        
        print("\n>>> Renderizando gráficos: Eje X (Original) vs Eje Y (Normalizado)...")
        with tqdm(total=total_graficos, desc="Progreso Gráficos", unit="img", colour='cyan') as pbar:
            # 1. Gráficos Originales
            for variable in cols_num:
                self.graficar_dispersion(etiquetas, df_original[variable].values, variable, "Original", es_comparativo=False)
                pbar.update(1)

            # 2. Gráficos Normalizados
            for nombre_metodo, df_norm in dict_calculos.items():
                for variable in cols_num:
                    vals_orig = df_original[variable].values      # Abscisas
                    vals_norm = df_norm[variable].values          # Ordenadas
                    self.graficar_dispersion(vals_orig, vals_norm, variable, nombre_metodo, 
                                             es_comparativo=True, etiquetas_puntos=etiquetas)
                    pbar.update(1)

    def graficar_dispersion(self, x_data, y_data, var_name, metodo_name, es_comparativo=False, etiquetas_puntos=None):
        """Dibuja los gráficos según los ejes solicitados."""
        plt.figure(figsize=(10, 6))
        
        if metodo_name == "Original":
            ruta_carpeta = self.dir_originales
            color_puntos = 'darkorange'
            # Eje X son etiquetas, Eje Y son valores
            plt.scatter(range(len(x_data)), y_data, s=150, c=color_puntos, alpha=0.6, edgecolors='black', zorder=3)
            plt.xticks(range(len(x_data)), x_data, rotation=35)
            plt.xlabel("Alternativas / Sucursales", fontsize=10)
            plt.ylabel(f"Valor Observado", fontsize=10)
        else:
            ruta_carpeta = self.dir_normalizados
            color_puntos = 'navy'
            # Eje X = Valor Original, Eje Y = Puntaje Normalizado
            plt.scatter(x_data, y_data, s=150, c=color_puntos, alpha=0.6, edgecolors='black', zorder=3)
            plt.xlabel(f"Valor Original)", fontsize=10)
            plt.ylabel("Valor Normalizado", fontsize=10)
        
        # Anotaciones
        if etiquetas_puntos is not None and es_comparativo:
            for i, txt in enumerate(etiquetas_puntos):
                plt.annotate(f"{txt}\n({y_data[i]:.2f})", (x_data[i], y_data[i]), 
                             textcoords="offset points", xytext=(0,10), ha='center', 
                             fontsize=8, fontweight='bold')
        elif metodo_name == "Original":
            for i, val in enumerate(y_data):
                plt.annotate(f"{val:.2f}", (i, y_data[i]), textcoords="offset points", 
                             xytext=(0,10), ha='center', fontsize=8, fontweight='bold')

        plt.title(f"{metodo_name.upper()} - {var_name}", fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        filename = f"{ruta_carpeta}/{metodo_name.replace(' ', '_')}_{var_name}.png"
        plt.savefig(filename, dpi=150)
        plt.close()

    def exportar_a_excel(self, resultados, nombre_archivo="Reporte_Final_Multicriterio.xlsx"):
        """Exporta las matrices resultantes a Excel."""
        print(f"\n>>> Guardando reporte en Excel...")
        with pd.ExcelWriter(nombre_archivo, engine='xlsxwriter') as writer:
            for nombre_metodo, df_res in resultados.items():
                sheet_name = nombre_metodo.replace(" ", "_")[:31]
                df_res.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"✅ Proceso Finalizado.\n📁 Carpetas: [{self.dir_originales}] y [{self.dir_normalizados}]")

    def ejecutar_todo(self, df, minimo=None, metas_rim=None, n_intervalos_oecd=5):
        """Ejecuta el flujo completo de normalización."""
        if minimo is None: minimo = []
        
        cols_num = df.select_dtypes(include=['number']).columns
        df_original = df[cols_num].copy()
        df_no_num = df.select_dtypes(exclude=['number']).reset_index(drop=True)
        etiquetas = df.iloc[:, 0].astype(str).tolist()

        print("\n" + "="*80 + "\n INICIANDO SISTEMA DE NORMALIZACIÓN MULTICRITERIO 2026 \n" + "="*80)

        self._limpiar_directorios()

        # 1. CÁLCULO
        print("\n>>> Ejecutando algoritmos de normalización...")
        dict_calculos = self.calcular_normalizaciones(df, cols_num, minimo, metas_rim, n_intervalos_oecd)
        
        resultados_finales = {}
        for nombre, df_calc in dict_calculos.items():
            resultados_finales[nombre] = pd.concat([df_no_num, df_calc.reset_index(drop=True)], axis=1)

        # 2. GRÁFICOS
        self.generar_reporte_visual(df_original, dict_calculos, etiquetas, cols_num)

        # 3. REPORTE CONSOLA
        for nombre, tabla in resultados_finales.items():
            print(f"\n>>> MATRIZ NORMALIZADA: {nombre.upper()}\n" + "-"*60)
            print(tabla.round(4).to_string(index=False, justify='center'))

        # 4. EXPORTACIÓN
        self.exportar_a_excel(resultados_finales)

        return resultados_finales
    
# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    data = {
        'Sucursal': ['Norte', 'Sur', 'Este', 'Oeste', 'Centro'],
        'Ventas_USD': [50000, 120000, 80000, 45000, 95000],
        'Gastos_USD': [12000, 25000, 18000, 9000, 20000],
        'Quejas_Num': [2, 10, 5, 1, 8]
        }
    df_sucursales = pd.DataFrame(data)

    metas_estrategicas = {
        'Ventas_USD': [0, 200000, 110000, 140000], # Meta: vender entre 110k y 140k
        'Gastos_USD': [5000, 30000, 8000, 13000],     # Meta: gastar entre 8k y 13k
        'Quejas_Num': [1, 20, 1, 2]                   # Meta: tener entre 0 y 2 quejas
    }

    motor = ModelosNormalizacion()

    # AQUÍ PUEDES DEFINIR EL NÚMERO DE INTERVALOS (ej. 3 para Bajo/Medio/Alto, o 10 para Deciles)
    resultados_finales = motor.ejecutar_todo(
        df_sucursales, 
        minimo=['Gastos_USD', 'Quejas_Num'], 
        metas_rim=metas_estrategicas,
        n_intervalos_oecd=2  # <--- Cambia este número aquí
    )
    
    dict_resultados = motor.calcular_normalizaciones(
        df=df_sucursales, 
        cols_num=df_sucursales.select_dtypes(include=['number']).columns, 
        minimo=['Gastos_USD', 'Quejas_Num'], 
        metas_rim=metas_estrategicas, 
        n_intervalos_oecd=5
    ) 
    
    print(dict_resultados['Ideal de Referencia'])
