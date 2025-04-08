# 🧠 Optimización Evolutiva de Redes Neuronales para MNIST

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

Algoritmo Genético para optimizar los pesos de una red neuronal en la clasificación de dígitos manuscritos (MNIST)

![Arquitectura de la Red](network_architecture.png)
*Arquitectura de la red neuronal 784-25-10*

## 📌 Características Principales

- **Algoritmo Evolutivo**: Optimización de pesos mediante estrategias genéticas
- **Paralelización**: Uso de Joblib para evaluación de aptitud en múltiples núcleos
- **Regularización L2**: Prevención de sobreajuste con coeficiente λ = 0.0001
- **Visualización Interactiva**: Gráficos de progreso y predicciones
- **MNIST Benchmark**: 85.4% de precisión en conjunto de prueba

## 📋 Tabla de Contenidos
- [Instalación](#🔧-instalación)
- [Resultados](#📊-resultados)
- [Tecnologías](#🛠️-tecnologías)
- [Contribución](#🤝-contribución)
- [Licencia](#📄-licencia)

## 🔧 Instalación

### Requisitos Previos
- Python 3.8 o superior.
- Gestor de paquetes `pip`.

### Pasos para Instalar
1. Clona el repositorio:
```bash
git clone https://github.com/tuusuario/optimizacion-redes-geneticas.git
cd optimizacion-redes-geneticas
```
2. Instala las dependencias:
```bash 
pip install -r requirements.txt
```
## 📊 Resultados
![Progreso del entrenamiento](loss_progress.png)
*Pérdida durante el entrenamiento*

![Ejemplos de Predicciones](predictions.png)
*Predicciones en MNIST*

Métricas Finales
| Métrica              | Valor     |
|----------------------|-----------|
| Mejor pérdida        | 0.4521    |
| Precisión en prueba  | 85.4%     |
| Tiempo de ejecución  | 2h 15m    |

## 🛠️ Tecnologías

- ![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
- ![NumPy](https://img.shields.io/badge/NumPy-1.26.4-purple?logo=numpy&logoColor=white)
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.3-orange?logo=matplotlib&logoColor=white)
- ![Joblib](https://img.shields.io/badge/Joblib-1.4.2-green)
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?logo=tensorflow&logoColor=white)

## 🤝 Contribución
1. **Haz un fork** del proyecto.
2. Crea una rama:
```bash
git checkout -b feature/nueva-funcionalidad
```
3.Realiza el commit de tus cambios:
```bash
git commit -m 'Añade funcionalidad X'
```
4. Haz push a la rama:
```bash
git push origin feature/nueva-funcionalidad
```
5. Abre un **Pull Request** detallando los cambios

## 📄 Licencia
Distribuido bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

## 📧 Contacto

- **Autor:** Anibal Medina Cabrera  
- **Correo:** [anibal.medina@uaem.edu.mx](mailto:anibal.medina@uaem.edu.mx)  
- **Twitter:** [@l3ect3er](https://twitter.com/l3ect3er)
