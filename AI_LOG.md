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

## Entrada 4 — Wrapper de PyTorch para dataset de Hugging Face
**Fecha:** 16 de marzo, 2026  
**Herramienta:** Claude (Anthropic)

**Prompt utilizado:**
> "¿Cómo cargo el dataset de Hugging Face como tensor de PyTorch? ¿No puedo simplemente usar ToTensor directo?"

**Respuesta resumida:**  
Claude explicó que Hugging Face y PyTorch son librerías distintas con estructuras de datos incompatibles. Para usar el dataset con DataLoader se necesita un wrapper que herede de `torch.utils.data.Dataset` e implemente `__len__` y `__getitem__`. Esto actúa como puente entre ambas librerías.

**Decisión tomada:**  
Se implementó la clase `GTSRBDataset` como wrapper. El transform se aplica dentro del `__getitem__` convirtiendo cada imagen PIL a tensor normalizado en el momento de acceso.

**Reflexión crítica:**  
El wrapper es básicamente un traductor entre dos ecosistemas. No es algo específico del GTSRB, es el patrón estándar de PyTorch para cualquier dataset externo. Una vez entendido el por qué, tiene sentido. La referencia en la documentación oficial es `torch.utils.data.Dataset`: https://pytorch.org/docs/stable/data.html

---

## Entrada 5 — Normalización: qué es y de dónde vienen los valores
**Fecha:** 16 de marzo, 2026  
**Herramienta:** Claude (Anthropic)

**Prompt utilizado:**
> "¿Para qué sirve el Normalize en los transforms? ¿De dónde salen esos valores de mean y std?"

**Respuesta resumida:**  
Claude explicó que Normalize centra los valores de los píxeles cerca de 0 después del ToTensor (que los lleva al rango [0,1]). Esto hace que el entrenamiento converja más rápido y sea más estable. Los valores de mean y std usados (`[0.3337, 0.3064, 0.3171]` y `[0.2672, 0.2564, 0.2629]`) fueron precalculados sobre el dataset GTSRB original por la comunidad.

**Decisión tomada:**  
Se usaron los valores precalculados como punto de partida por ser los estándar del dataset. Se aplicó solo a val y test con `transform`, y al train con `transform_train` que incluye augmentation adicional.

**Reflexión crítica:**  
Queda la duda de si usar valores estáticos calculados sobre otro subset es lo correcto. En producción, lo ideal sería calcular mean y std sobre el propio dataset de entrenamiento, ya que en un entorno real las condiciones de las imágenes siempre cambian. Para este proyecto los valores precalculados son aceptables, pero es una limitación a documentar en el análisis final.

---

## Entrada 6 — Arquitectura del modelo: estilo parametrizable vs hardcoded
**Fecha:** 21 de marzo, 2026  
**Herramienta:** Claude (Anthropic)

**Prompt utilizado:**
> "¿Puedo organizar el modelo de otra manera? Quiero algo más parecido al estilo del curso de mrdbourke donde los bloques están separados y los parámetros se pasan desde afuera."

**Respuesta resumida:**  
Claude propuso inicialmente una arquitectura con `nn.Sequential` monolítico. Al mostrar el estilo del curso de mrdbourke con bloques separados (`conv_block_1`, `conv_block_2`, etc.) y parámetros como `input_shape`, `hidden_units` y `output_shape`, Claude reconoció que es mejor enfoque y adaptó la arquitectura a ese estilo.

**Decisión tomada:**  
Se adoptó el estilo parametrizable con bloques separados. La clase `CNNBaseline` recibe `input_shape=3`, `hidden_units=32` y `output_shape=43`. Si se quiere cambiar la profundidad o el número de filtros, se modifica en un solo lugar al instanciar el modelo.

**Reflexión crítica:**  
El estilo parametrizable es objetivamente mejor para experimentación: cambiar `hidden_units` de 32 a 64 afecta toda la red automáticamente sin tocar la arquitectura. La propuesta inicial de Claude era funcional pero menos flexible. La decisión de cuestionar el primer enfoque y compararlo con lo aprendido en el curso fue correcta.

---

## Entrada 7 — Optimizador: Adam vs SGD
**Fecha:** 21 de marzo, 2026  
**Herramienta:** Claude (Anthropic)

**Prompt utilizado:**
> "¿Puedo usar SGD en lugar de Adam?"

**Respuesta resumida:**  
Claude recomendó Adam sobre SGD para el baseline porque converge más rápido y requiere menos ajuste de hiperparámetros. SGD necesita momentum y learning rate bien calibrados para funcionar bien en CNNs, lo cual se presta más para la semana de optimización.

**Decisión tomada:**  
Se usó Adam con `lr=0.001`. Se dejó SGD como candidato a explorar en semana 6 cuando toca ajuste de hiperparámetros.

**Reflexión crítica:**  
La sugerencia fue razonable. Adam es el estándar para prototipos rápidos. Sin embargo, hay literatura que muestra que SGD con momentum bien calibrado puede superar a Adam en generalización. Queda pendiente compararlo experimentalmente en semana 6.

---

## Entrada 8 — Uso de GPU y DataLoader workers
**Fecha:** 21 de marzo, 2026  
**Herramienta:** Claude (Anthropic)

**Prompt utilizado:**
> "¿Por qué la GPU solo está al 6% si estoy entrenando con CUDA?"

**Respuesta resumida:**  
Claude explicó que el cuello de botella era el CPU cargando datos más lento de lo que la GPU puede entrenar. La solución es agregar `num_workers=4` y `pin_memory=True` al DataLoader para que el CPU prepare batches en paralelo mientras la GPU entrena.

**Decisión tomada:**  
Se actualizaron los tres DataLoaders con `num_workers=4` y `pin_memory=True`. El tiempo de entrenamiento bajó de 172 segundos a 45 segundos (casi 4x más rápido) con `batch_size=64`.

**Reflexión crítica:**  
El resultado fue contundente: de 17s/it a 4.7s/it. Se probaron batch sizes de 32 y 64. Con 64 el tiempo bajó drásticamente sin afectar el accuracy (98.56% vs 98.62%). Con imágenes pequeñas de 32x32 la GPU sigue sin saturarse al 100% porque los batches se procesan muy rápido. Para saturarla se necesitarían imágenes más grandes o un modelo más complejo.

---
