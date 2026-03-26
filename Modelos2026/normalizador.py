# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from tqdm import tqdm

# Configuración de estilo de gráficos para reportes gerenciales
sns.set_theme(style="whitegrid")

class ModelosNormalizacion:
    def __init__(self):
        # Simplificamos a una sola carpeta de salida para mejor orden
        self.dir_reportes = "Reportes_Visuales_2026"
        self._crear_directorios()

    def _crear_directorios(self):
        if not os.path.exists(self.dir_reportes):
            os.makedirs(self.dir_reportes)

    def _limpiar_directorios(self):
        """Borra archivos antiguos para evitar confusión de resultados."""
        if os.path.exists(self.dir_reportes):
            shutil.rmtree(self.dir_reportes)
        os.makedirs(self.dir_reportes)
        print(">>> Directorio de reportes listo (limpio).")

    # --- (Tus funciones matemáticas originales se mantienen igual) ---
    def _aplicar_reciproca(self, df, columnas):
        df_temp = df.copy()
        for col in columnas:
            if col in df_temp.columns:
                # Evitamos división por cero con un valor muy pequeño
                df_temp[col] = 1 / df_temp[col].replace(0, 1e-9)
        return df_temp

    def normalizacion_oecd_pro(self, df_procesado, columnas_num, n_intervalos=5):
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
        # Aplicamos recíproca a los criterios de 'costo' (minimizar)
        df_p = self._aplicar_reciproca(df, minimo)
        
        metodos = {
            'Fraccion Rango': (df_p[cols_num] - df_p[cols_num].min()) / (df_p[cols_num].max() - df_p[cols_num].min() + 1e-9),
            'Fraccion Suma': df_p[cols_num] / (df_p[cols_num].sum() + 1e-9),
            'Fraccion Maximo': df_p[cols_num] / (df_p[cols_num].max() + 1e-9),
            'Fraccion Modulo': df_p[cols_num] / np.sqrt((df_p[cols_num]**2).sum() + 1e-9),
            'Z-Score': (df_p[cols_num] - df_p[cols_num].mean()) / (df_p[cols_num].std() + 1e-9),
            'Categorica': self.normalizacion_oecd_pro(df_p, cols_num, n_intervalos=n_intervalos_oecd),
            'Ideal de Referencia': self.normalizacion_rim_pro(df[cols_num], cols_num, metas_rim)
        }
        return metodos
    # -----------------------------------------------------------------

    # --- SECCIÓN MODIFICADA: GENERACIÓN DEL REPORTE VISUAL COMPARATIVO ---

    def generar_reporte_visual(self, df_original, dict_calculos, etiquetas, cols_num):
        """Genera gráficos comparativos Valor Real vs. Puntaje Normalizado."""
        # Calculamos el total de gráficos: métodos x variables
        total_graficos = len(dict_calculos) * len(cols_num)
        
        print(f"\n>>> Renderizando {total_graficos} gráficos comparativos...")
        with tqdm(total=total_graficos, desc="Progreso Gráficos", unit="img", colour='cyan') as pbar:
            for nombre_metodo, df_norm in dict_calculos.items():
                for variable in cols_num:
                    self.graficar_comparacion(
                        etiquetas=etiquetas,
                        valores_origen=df_original[variable].values,
                        valores_norm=df_norm[variable].values,
                        var_name=variable,
                        metodo_name=nombre_metodo
                    )
                    pbar.update(1)

    def graficar_comparacion(self, etiquetas, valores_origen, valores_norm, var_name, metodo_name):
        """Gráfico de dispersión: Valor Real (Eje X) vs. Puntaje (Eje Y)."""
        
        # 1. Configuración de la figura
        plt.figure(figsize=(11, 7))
        
        # 2. Renderizado de puntos (X=Origen, Y=Normalizado)
        # Usamos un mapa de colores para dar profundidad visual
        scatter = plt.scatter(valores_origen, valores_norm, s=250, c=valores_norm, 
                              cmap='viridis', alpha=0.7, edgecolors='black', linewidths=1.5, zorder=3)
        
        # 3. Etiquetas de las alternativas (ej: nombres de proyectos)
        for i, txt in enumerate(etiquetas):
            # Desplazamos ligeramente el texto para que no tape el punto
            plt.annotate(txt, (valores_origen[i], valores_norm[i]), 
                         textcoords="offset points", xytext=(0,12), ha='center', 
                         fontsize=8, fontweight='bold', color='black')

        # 4. Diseño del Reporte Gerencial
        plt.title(f"EFECTO DE NORMALIZACIÓN: {metodo_name.upper()}\nVariable: {var_name}", 
                  fontsize=14, fontweight='bold', pad=20)
        
        # Explicación clara de los ejes para el alumno
        plt.xlabel(f"Valor Original de la Variable ({var_name})", fontsize=11, fontweight='bold')
        plt.ylabel(f"Puntaje Normalizado (Escala Comparativa)", fontsize=11, fontweight='bold')
        
        # Añadimos barra de color para referencia visual
        cbar = plt.colorbar(scatter)
        cbar.set_label('Intensidad del Puntaje', rotation=270, labelpad=15)

        # Mejoras estéticas
        plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
        
        # Ajuste de límites para que las etiquetas no se corten
        margen_x = (max(valores_origen) - min(valores_origen)) * 0.1 if len(valores_origen)>1 else 1
        plt.xlim(min(valores_origen) - margen_x, max(valores_origen) + margen_x)
        
        # Forzamos los ticks en X para que coincidan con los valores reales si son pocos
        if len(valores_origen) < 10:
            plt.xticks(valores_origen)

        plt.tight_layout()
        
        # 5. Guardado del archivo
        # Reemplazamos espacios por guiones bajos para nombres de archivo compatibles
        metodo_safe = metodo_name.replace(' ', '_')
        filename = f"{self.dir_reportes}/Comp_{metodo_safe}_{var_name}.png"
        plt.savefig(filename, dpi=150) # 150 dpi es suficiente para reportes
        plt.close()

    # -------------------------------------------------------------------------

    def exportar_a_excel(self, resultados, nombre_archivo="Reporte_Final_Multicriterio.xlsx"):
        print(f"\n>>> Guardando reporte numérico en Excel...")
        #xlsxwriter permite formatos profesionales
        with pd.ExcelWriter(nombre_archivo, engine='xlsxwriter') as writer:
            for nombre_metodo, df_res in resultados.items():
                #xlsxwriter limita el nombre de la hoja a 31 caracteres
                sheet_name = nombre_metodo.replace(" ", "_")[:31]
                df_res.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"✅ Proceso Finalizado.\n📁 Reportes visuales en: [{self.dir_reportes}]\n📄 Reporte Excel: [{nombre_archivo}]")

    def ejecutar_todo(self, df, minimo=None, metas_rim=None, n_intervalos_oecd=5):
        """Flujo integral de ejecución para alumnos."""
        if minimo is None: minimo = []
        
        # Separación de datos numéricos y categóricos
        cols_num = df.select_dtypes(include=['number']).columns
        df_original = df[cols_num].copy()
        df_no_num = df.select_dtypes(exclude=['number']).reset_index(drop=True)
        # Usamos la primera columna como etiquetas de alternativas
        etiquetas = df.iloc[:, 0].astype(str).tolist()

        print("\n" + "="*80 + "\n INICIANDO SISTEMA DE NORMALIZACIÓN MULTICRITERIO 2026 \n" + "="*80)

        self._limpiar_directorios()

        # 1. CÁLCULO
        print("\n>>> Ejecutando algoritmos de normalización...")
        dict_calculos = self.calcular_normalizaciones(df, cols_num, minimo, metas_rim, n_intervalos_oecd)
        
        # Reconstruimos las tablas finales uniendo datos categóricos
        resultados_finales = {}
        for nombre, df_calc in dict_calculos.items():
            resultados_finales[nombre] = pd.concat([df_no_num, df_calc.reset_index(drop=True)], axis=1)

        # 2. GRÁFICOS (Sección Modificada)
        self.generar_reporte_visual(df_original, dict_calculos, etiquetas, cols_num)

        # 3. CONSOLA (Opcional, muestra las primeras filas)
        for nombre, tabla in resultados_finales.items():
            print(f"\n>>> MATRIZ NORMALIZADA: {nombre.upper()} (Top 5)\n" + "-"*60)
            print(tabla.round(4).head().to_string(index=False, justify='center'))

        # 4. EXCEL
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
