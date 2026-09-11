# NLP_taller_1

# Sistema Híbrido de Reconocimiento de Entidades y Modelado Secuencial para Textos Clínicos en Español.

## Integrantes

**Carlos Javier Cepeda, David Salamanca, Jose Milciades Ordoñez** 

## Objetivo: 

El objetivo es construir un modelo de Procesamiento de Lenguaje Natural (NLP) que no solo procese palabras aisladas, sino que entienda el contexto y la terminología médica especializada para extraer información crítica (síntomas, medicamentos, patologías) de historias clínicas o diagnósticos.

## Definición del Modelo

El núcleo del proyecto es una arquitectura que fusiona conocimiento lingüístico simbólico (reglas y modelos estadísticos tradicionales) con aprendizaje profundo (Deep Learning).

En lugar de que la red neuronal empiece a aprender desde cero qué es un verbo o un sustantivo, le entregaremos el texto previamente analizado. El modelo se encargará de resolver tareas de etiquetado de secuencias (Sequence Labeling), donde la red recibe una oración médica y debe predecir, token por token, a qué categoría clínica pertenece (por ejemplo, si la palabra "paracetamol" es un <FÁRMACO> o "cefalea" es un <SÍNTOMA>).

Para lograr esto como trabajo académico, el sistema se basa en dos pilares:

- spaCy (El extractor de características): Actúa como el motor de preprocesamiento industrial. Analiza la morfología, extrae las raíces de las palabras (lemas) y detecta la estructura gramatical básica (POS tagging).

- Red LSTM (El motor de contexto): Actúa como el cerebro secuencial. Las redes con Memoria a Largo Plazo (LSTM) son ideales aquí porque pueden "recordar" que un síntoma mencionado al inicio de un párrafo largo está directamente relacionado con un diagnóstico al final del mismo, mitigando el problema del desvanecimiento del gradiente.

