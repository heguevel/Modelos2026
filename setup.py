# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 04:54:33 2026

@author: hegue
"""

from setuptools import setup, find_packages

setup(
    name="Modelos2026",
    version="1.0.0",
    author="Tu Nombre",
    description="Paquete de modelos para alumnos de administración",
    packages=find_packages(), # Busca automáticamente la carpeta Modelos2026
    install_requires=[
        "pandas", 
        "numpy", 
        "matplotlib"
    ], # Librerías que el alumno necesita tener
)