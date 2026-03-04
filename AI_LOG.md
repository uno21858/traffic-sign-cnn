# AI_LOG.md — Bitácora de Uso de IA
**Proyecto:** Clasificación de Señales de Tráfico con CNN  
**Estudiante:** Erick Alberto Sánchez Aranda | A01641715  
**Dataset:** GTSRB — German Traffic Sign Recognition Benchmark

---

## Entrada 1 — Selección del Dataset
**Fecha:** 28 de febrero, 2026  
**Herramienta:** Claude (Anthropic)

**Prompt utilizado:**
> "Recomiéndame un dataset para un proyecto de CNN. Busco algo realista, que no sea trivial, y que tenga relevancia para empresas como NVIDIA o en el área de visión computacional. No quiero datasets médicos."

**Respuesta resumida:**  
Claude recomendó el dataset GTSRB (German Traffic Sign Recognition Benchmark), disponible en Hugging Face. Argumentó que es relevante para vehículos autónomos y robótica, que tiene más de 50,000 imágenes con 43 clases, y que los datos son suficientemente sucios (variaciones de iluminación, ángulos, oclusiones) para que el problema sea real y no de juguete.

**Decisión tomada:**  
Se aceptó la recomendación. El dataset coincide con los objetivos personales del proyecto: aprender CNN con un caso de uso directamente aplicable en empresas de visión computacional como NVIDIA o en sistemas de conducción autónoma.

**Reflexión crítica:**  
La sugerencia fue acertada. Claude descartó datasets médicos por el riesgo de daño real ante falsos positivos, lo cual coincide con mi propio criterio: no quería asumir responsabilidad sobre un sistema de diagnóstico que pudiera afectar decisiones clínicas reales (referencia al caso Therac-25). El argumento de relevancia para NVIDIA fue el factor decisivo. No se modificó la sugerencia.

---

## Entrada 2 — Justificación Técnica del Dataset
**Fecha:** 28 de febrero, 2026  
**Herramienta:** Claude (Anthropic)

**Prompt utilizado:**
> "Ayúdame a redactar la justificación técnica del dataset para la semana 1. Quiero que quede claro por qué elegí señales de tráfico y no otras opciones como enfermedades, plantas o basura."

**Respuesta resumida:**  
Claude generó una justificación estructurada que incluye: descripción del dataset, descarte razonado de otras alternativas, relevancia en vehículos autónomos y robótica, y la conexión con empresas objetivo (NVIDIA, Tesla, Waymo). También incluyó una propuesta de métricas de evaluación.

**Decisión tomada:**  
Se tomó la estructura general propuesta pero se ajustó el tono y se agregaron los recursos consultados previamente (curso de mrdbourke, CNN Explainer, 3Blue1Brown) como referencias reales de donde se obtuvo el conocimiento base para la propuesta.

**Reflexión crítica:**  
El documento generado por IA tenía un tono demasiado formal en algunos párrafos. Se revisó para que reflejara el razonamiento propio, especialmente en la sección de descarte de datasets médicos donde el argumento personal (Therac-25) es más honesto que una justificación genérica. La IA fue útil para estructurar, no para pensar.

---

## Entrada 3 — Propuesta de Arquitectura CNN Inicial
**Fecha:** 28 de febrero, 2026  
**Herramienta:** Claude (Anthropic)

**Prompt utilizado:**
> "Para el baseline de semana 3, ¿qué arquitectura CNN desde cero me recomiendas para clasificar 43 clases de señales de tráfico?"

**Respuesta resumida:**  
Claude propuso una arquitectura de 3 bloques convolucionales (32, 64 y 128 filtros con kernel 3x3 y MaxPooling), seguida de Flatten, Dropout (0.5), una capa densa de 256 neuronas y salida Softmax de 43 clases.

**Decisión tomada:**  
Se aceptó como punto de partida para el baseline. Esta arquitectura se comparará contra un modelo preentrenado (ResNet o EfficientNet) en la semana 5.

**Reflexión crítica:**  
La arquitectura propuesta es estándar y conservadora, lo cual es correcto para un baseline: no tiene sentido sobrecomplicar el punto de partida. Sin embargo, no se validó experimentalmente aún. La arquitectura podría necesitar ajustes dependiendo del tamaño de entrada de las imágenes del GTSRB (que varía entre 15x15 y 250x250 píxeles). Pendiente definir el tamaño de resize estándar antes de implementar (probablemente 32x32 o 64x64).

---
