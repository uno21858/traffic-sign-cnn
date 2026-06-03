# Traffic Sign CNN Classifier

Sistema de clasificación de señales de tráfico usando Redes Neuronales Convolucionales, entrenado sobre el dataset GTSRB (German Traffic Sign Recognition Benchmark).

**Demo en vivo:** https://cnn.uno21things.dev

## Video Demo

[![Demo](https://img.youtube.com/vi/D6qYXlD6pt4/0.jpg)](https://youtu.be/D6qYXlD6pt4)

---

## Descripción

Proyecto desarrollado como Servicio Becario en el Área de Inteligencia Artificial del Tecnológico de Monterrey. Implementa un pipeline completo de clasificación de imágenes: desde el preprocesamiento del dataset hasta un servidor REST con interfaz web funcional.

- **43 clases** de señales de tráfico alemanas
- **92% de accuracy** sobre el test set oficial (12,630 imágenes)
- **API REST** con FastAPI, accesible via Cloudflare Tunnel
- **UI minimalista** para clasificar imágenes en tiempo real

---

## Dataset

**GTSRB — German Traffic Sign Recognition Benchmark**  
Disponible en Hugging Face: https://huggingface.co/datasets/tanganke/gtsrb

| Split | Imágenes |
|-------|----------|
| Train | 26,640 |
| Validación (80/20 del train) | 5,328 |
| Test oficial | 12,630 |

---

## Arquitectura

### CNNBaseline (modelo final)

```
Input: [batch, 3, 32, 32]
→ Conv2D(3→32, 3x3) + ReLU + MaxPool2D(2x2)
→ Conv2D(32→64, 3x3) + ReLU + MaxPool2D(2x2)
→ Conv2D(64→128, 3x3) + ReLU + MaxPool2D(2x2)
→ Flatten + Dropout(0.5)
→ Linear(2048→256) + ReLU
→ Linear(256→43)
```

- **Parámetros totales:** 628,843
- **Framework:** PyTorch 2.10.0 + CUDA 12.8
- **Hardware:** NVIDIA GeForce RTX 3060 (12GB VRAM)

### Transfer Learning (experimental)

Se implementó fine-tuning de ResNet18 preentrenado en ImageNet descongelando `layer4 + fc`. Alcanzó 97.92% en validación pero presentó domain shift en el test oficial. Ver reporte de semana 5.

---

## Resultados

| Modelo | Accuracy (test) | F1 weighted | Tiempo (10 epochs) |
|--------|----------------|-------------|-------------------|
| CNNBaseline | 92% | 0.92 | 45s |
| ResNet18 fine-tuning | 76%* | 0.75 | 53s |

*La diferencia se debe a domain shift entre los splits del dataset. Ver `training/semana7_analisis_critico.docx`.

---

## Estructura del Proyecto

```
traffic-sign-cnn/
├── data_prep/
│   └── preprocessing.ipynb       # Carga, transforms, splits, DataLoaders
├── training/
│   ├── baseline.ipynb            # CNN desde cero, entrenamiento
│   ├── evaluation.ipynb          # Métricas, matriz de confusión
│   └── transfer_learning.ipynb   # ResNet18 fine-tuning
├── inference/
│   └── predict.py                # Función predict(image_path)
├── models/                       # Pesos .pth (no incluidos en repo)
├── server/
│   └── main.py                   # FastAPI con endpoint /predict
├── ui/
│   └── index.html                # Interfaz web
├── Dockerfile
├── requirements.txt
├── README.md
├── DEPLOYMENT.md
└── AI_LOG.md
```

---

## Correr el servidor localmente

### Con Docker (recomendado)

```bash
# 1. Clonar el repo
git clone https://github.com/uno21858/traffic-sign-cnn
cd traffic-sign-cnn

# 2. Descargar el modelo entrenado
# Coloca baseline_cnn.pth en /home/tu-usuario/models/

# 3. Construir y correr
docker build -t traffic-api .
docker run -d \
  --name traffic-api \
  --gpus device=0 \
  -p 8000:8000 \
  --restart always \
  -v /home/tu-usuario/models:/home/uno21/models \
  traffic-api
```

### Sin Docker

```bash
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

La API queda disponible en `http://localhost:8000`.  
La UI en `http://localhost:8000`.  
Documentación interactiva en `http://localhost:8000/docs`.

---

## Modelo entrenado

El modelo no está incluido en el repositorio por su tamaño. Para obtenerlo contacta al autor o entrena desde cero con `training/baseline.ipynb`.

---

## Uso de IA

Todo el uso de herramientas de IA generativa durante el proyecto está documentado en [`AI_LOG.md`](./AI_LOG.md).

---

**Estudiante:** Erick Alberto Sánchez Aranda · A01641715  
**Institución:** Tecnológico de Monterrey, Guadalajara  
**Área:** Inteligencia Artificial · Servicio Becario
