# Procesamiento de Lenguaje Natural — Talleres

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![spaCy](https://img.shields.io/badge/spaCy-3.8-09a3d5)
![Status](https://img.shields.io/badge/Estado-en%20desarrollo-yellow)
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jose-milciades/NLP_taller_1/blob/main/notebook_taller_1.ipynb)

Repositorio del curso de Maestría en **Procesamiento de Lenguaje Natural**. Reúne la solución de los distintos talleres de la materia, todos construidos sobre el **mismo corpus clínico en español**, pero explorando **modelos, arquitecturas, hiperparámetros y bibliotecas diferentes** en cada entrega. Esto permite comparar, taller a taller, distintos enfoques frente a un mismo problema de referencia.

## Tabla de contenido

- [Integrantes](#integrantes)
- [Dataset](#dataset)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Taller 1 — NER híbrido (spaCy + Bi-LSTM)](#taller-1--sistema-híbrido-de-reconocimiento-de-entidades-spacy--bi-lstm)
- [Cómo ejecutar los cuadernos](#cómo-ejecutar-los-cuadernos)
- [Licencia](#licencia)

## Integrantes

**Carlos Javier Cepeda, David Salamanca, Jose Milciades Ordoñez**

## Dataset

Todos los talleres usan el corpus clínico **SPACCC (Spanish Clinical Case Corpus)**, publicado en Hugging Face:

| Recurso | Contenido |
|---|---|
| [`IEETA/SPACCC-Spanish-NER`](https://huggingface.co/datasets/IEETA/SPACCC-Spanish-NER) | Anotaciones NER: 750 documentos de entrenamiento (33,757 anotaciones) y 250 de test (11,239 anotaciones) |
| [`IEETA/SPACCC-documents`](https://huggingface.co/datasets/IEETA/SPACCC-documents) | Texto completo de cada historia clínica |

Categorías anotadas: `CHEMICAL`, `DISEASE`, `PROCEDURE`, `PROTEIN`, `SYMPTOM`.

## Estructura del repositorio

```
NLP_taller_1/
├── README.md
└── notebook_taller_1.ipynb   # Taller 1: NER híbrido spaCy + Bi-LSTM
```

Cada taller nuevo se documenta en una sección propia de este README y, cuando aplique, en su propio cuaderno (`notebook_taller_N.ipynb`) o carpeta, manteniendo el mismo dataset como base de comparación.

## Taller 1 — Sistema Híbrido de Reconocimiento de Entidades (spaCy + Bi-LSTM)

**Cuaderno:** [`notebook_taller_1.ipynb`](notebook_taller_1.ipynb) · [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jose-milciades/NLP_taller_1/blob/main/notebook_taller_1.ipynb)

### Objetivo

Construir un modelo de NLP que no solo procese palabras aisladas, sino que entienda el contexto y la terminología médica especializada para extraer información crítica (síntomas, medicamentos, patologías) de historias clínicas.

### Arquitectura

El núcleo del taller es una arquitectura híbrida que fusiona conocimiento lingüístico simbólico con aprendizaje profundo, en lugar de dejar que la red aprenda gramática española desde cero:

- **spaCy** (`es_core_news_sm`) — extractor de características: tokeniza el texto y obtiene el POS (Part-of-Speech) de cada token.
- **Bi-LSTM** (PyTorch) — motor de contexto: red bidireccional que recibe los embeddings de palabra (y, opcionalmente, los de POS concatenados) y predice, token por token, la etiqueta BIO correspondiente.

Se entrenan y comparan dos variantes:

| Variante | Entrada al modelo |
|---|---|
| `con_pos` | Embedding de palabra + embedding de POS |
| `sin_pos` | Solo embedding de palabra (control experimental) |

### Metodología

El cuaderno está organizado en 12 pasos secuenciales, pensados para ejecutarse en Kaggle con GPU (2× Tesla T4):

1. Instalación de dependencias y modelo de spaCy.
2. Configuración, semillas y verificación de GPU.
3. Descarga y auditoría del corpus.
4. Partición fija por documento (`split_seed=42`): 600 train / 150 validación / 250 test reservado.
5. Tokenización, alineación de offsets y construcción de etiquetas BIO.
6. Vocabulario, `DataLoader` y función de pérdida ponderada por clase.
7. Definición de la Bi-LSTM y de las métricas de evaluación (coincidencia exacta de entidades).
8. Prueba técnica (smoke test) antes de entrenar.
9. Seis entrenamientos independientes: 2 variantes × 3 semillas (`42`, `123`, `2026`).
10. Agregación de resultados (media, desviación estándar, diferencias pareadas).
11. Evaluación final sobre el conjunto de test (opcional, deshabilitada por defecto).
12. Inferencia interactiva sobre texto nuevo.

### Resultados

Comparación multisemilla en validación (F1 en %):

| Variante | F1 medio | Desv. estándar | Precisión media | Recall medio |
|---|---:|---:|---:|---:|
| `con_pos` | **38.96 %** | 0.51 % | 32.44 % | 48.76 % |
| `sin_pos` | 37.13 % | 1.30 % | 30.90 % | 46.53 % |

La variante con POS obtuvo mayor F1 en las tres semillas evaluadas, con una mejora media de **+1.83 puntos porcentuales**. En la evaluación final sobre el conjunto de test reservado (`con_pos`, semilla 42) se obtuvo un F1 micro de **39.17 %** sobre 9,415 entidades.

### Limitaciones

- Esquema BIO plano: las anotaciones anidadas o solapadas se excluyen y se contabilizan, no se modelan.
- La confianza reportada en la inferencia interactiva no está calibrada.
- La comparación usa una única partición fija y tres semillas; la desviación estándar describe variabilidad de entrenamiento, no significancia estadística.
- Las predicciones son un ejercicio académico y no sustituyen una decisión clínica.

## Cómo ejecutar los cuadernos

1. Abrir el cuaderno correspondiente en un entorno con GPU e Internet habilitado:
   - **Kaggle** (entorno original, recomendado para reproducir los resultados reportados): acelerador **GPU T4 ×2**.
   - **Google Colab**: usa el botón *Abrir en Colab* de arriba, o el badge dentro del propio cuaderno. El plan gratuito de Colab entrega **una sola GPU T4**, así que antes de ejecutar el PASO 02 cambia `requested_gpus=1` (o `allow_cpu_for_debug=True` para depurar sin GPU); el cuaderno funciona igual, pero sin `DataParallel` y con tiempos de entrenamiento distintos a los reportados en el informe final.
2. Ejecutar las celdas en orden desde el PASO 01; cada paso valida sus propias precondiciones y detiene la ejecución con un mensaje claro si algo falta.
3. Los artefactos (checkpoints, métricas, `summary.json`, tablas `comparison_*.csv`) quedan guardados en una carpeta con marca de tiempo dentro de `spaccc_gpu/` (en Colab, típicamente bajo `/content/spaccc_gpu/`).

## Licencia

Proyecto académico desarrollado para el curso de Maestría en Procesamiento de Lenguaje Natural. El corpus SPACCC conserva la licencia de sus autores originales en Hugging Face.
