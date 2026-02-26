# Redes_comparacion_forense_voz
1. Descripción General
El  repositorio contiene la implementación de un sistema de verificación de locutor, diseñado para comparar pares de grabaciones de audio y determinar matemáticamente la probabilidad de que pertenezcan a la misma persona. El proyecto incluye múltiples iteraciones experimentales.

El código procesa la señal de audio, extrae vectores representativos (embeddings) y ejecuta comparaciones geométricas y estadísticas (Similitud Coseno, Likelihood Ratio) contra bases de datos referenciales.

2. Archivos Principales
El repositorio se divide en módulos de extracción, entrenamiento y evaluación:

Extracción y Modelado Clásico
Audio_Feature_Extraction.py: Módulo fundamental para el preprocesamiento de la señal. Contiene las funciones de transformación matemática, incluyendo inicialización de bancos de filtros y cálculo de coeficientes cepstrales de frecuencia de Mel (MFCC). 


app.py: Iteración basada en Modelos de Mezclas Gaussianas (GMM). Utiliza el módulo anterior para calcular probabilidades estadísticas locales en lugar de redes neuronales preentrenadas. 

Evaluaciones con Redes Neuronales
train.py: Script de entrenamiento desde cero. Genera el archivo speaker_nn.pt. 


app_nn1.py: Primera versión de inferencia neuronal. Implementa un FrameEncoder estático para procesar los MFCC y compara los audios utilizando distancia coseno directa sin calibración externa. Extrae las predicciones usando el archivo speaker_nn.pt. 


app_nn_v2.py: Integración de la arquitectura TDNN (X-Vector) mediante la librería SpeechBrain sobre VoxCeleb. Calcula similitudes basándose estrictamente en un mapeo de probabilidad (0 a 1) para estabilizar el cálculo del Likelihood Ratio. Exporta resultados masivos a formato Excel con base en listas de combinaciones predefinidas. 


app_nn_v3.py: Refinamiento del modelo X-Vector mediante la integración de una red Siamesa (ProyectorSiames). Esta arquitectura secundaria ajusta geométricamente los vectores extraídos para forzar la evaluación en la escala completa de Similitud Coseno [-1.0 a 1.0], requiriendo un archivo proyector_siames.pt generado localmente. 


Análisis Estadístico
resultados.py: Secuencia de comandos automatizada para consolidar múltiples documentos de salida (.xlsx). Mide el desempeño global (Exactitud, Precisión, Sensibilidad, F1-Score), calcula el umbral óptimo (EER) y genera un reporte técnico en texto plano junto con matrices de confusión y gráficos de distribución. 




Se recomienda aislar el entorno de trabajo:

Bash
python -m venv venv
Instale los paquetes especificados:

Bash
pip install -r requirements.txt
4. Configuración de Rutas de Trabajo
Antes de ejecutar los archivos de inferencia (app_nn1.py, app_nn_v2.py o app_nn_v3.py), es obligatorio abrir el código fuente y modificar los valores globales que apuntan a sus carpetas locales:


AUDIO_EVAL_PATH: Directorio que contiene los archivos dubitados que serán evaluados.



AUDIO_REF_PATH: Directorio que contiene la población general para el cálculo de tipicidad.



OUTPUT_DIR: Directorio destino de los documentos Excel.


5. Glosario de Salidas en Excel
Los scripts exportan los siguientes datos matemáticos para evaluación forense:


Similitud: Métrica geométrica entre los dos audios. (En app_nn_v2.py se presenta como un score mapeado; en app_nn_v3.py, abarca el espectro completo [-1, 1]).


Tipicidad: Mide el parecido promedio frente a voces aleatorias (Población de Referencia).


LR (Likelihood Ratio): Similitud probabilística dividida entre la Tipicidad.

LLR (Log-Likelihood Ratio): Logaritmo natural del LR. Usado para centrar estadísticamente los resultados.
