# DEPLOYMENT.md — Guía de Despliegue Reproducible

Este documento describe cómo reproducir el entorno completo del proyecto desde cero en una máquina nueva.

---

## Requisitos de Hardware

- GPU NVIDIA con soporte CUDA 12.x (probado en RTX 3060 12GB)
- 8GB RAM mínimo
- 10GB de espacio en disco

## Requisitos de Software

- Ubuntu 24.04 o RHEL 10 (probado en ambos)
- Docker 24+ con nvidia-container-toolkit
- Python 3.12
- Drivers NVIDIA 580+

---

## 1. Instalar nvidia-container-toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verifica:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## 2. Clonar el repositorio

```bash
git clone https://github.com/uno21858/traffic-sign-cnn
cd traffic-sign-cnn
```

---

## 3. Obtener el modelo entrenado

El modelo no está incluido en el repositorio. Tienes dos opciones:

### Opción A — Entrenar desde cero

```bash
# Instalar dependencias
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Correr el notebook de entrenamiento
jupyter lab training/baseline.ipynb
```

El notebook descarga el dataset automáticamente desde Hugging Face y entrena el modelo. Al finalizar guarda `baseline_cnn.pth`.

### Opción B — Descargar el modelo preentrenado

Contactar al autor para obtener el archivo `baseline_cnn.pth`.

Coloca el modelo en:

```bash
mkdir -p ~/models
cp baseline_cnn.pth ~/models/
```

---

## 4. Construir y correr el servidor

```bash
docker build -t traffic-api .

docker run -d \
  --name traffic-api \
  --gpus device=0 \
  -p 8000:8000 \
  --restart always \
  -v ~/models:/home/uno21/models \
  traffic-api
```

Verifica que el servidor esté corriendo:

```bash
curl http://localhost:8000
# Respuesta esperada: {"status":"ok","model":"CNNBaseline GTSRB","classes":43}
```

---

## 5. Probar el endpoint

```bash
# Descarga una imagen de prueba del dataset
python3 -c "
from datasets import load_dataset
ds = load_dataset('tanganke/gtsrb')
ds['test'][0]['image'].save('test_image.png')
print('Label real:', ds['test'][0]['label'])
"

# Clasifica la imagen
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.png"
```

---

## 6. Acceder a la UI

Abre en el browser:

```
http://localhost:8000
```

O si configuraste Cloudflare Tunnel:

```
https://cnn.uno21things.dev
```

La documentación interactiva de la API está en:

```
http://localhost:8000/docs
```

---

## 7. Exponer con Cloudflare Tunnel (opcional)

```bash
# Instalar cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.rpm -o cloudflared.rpm
sudo rpm -i cloudflared.rpm

# Autenticar
cloudflared tunnel login

# Crear tunnel
cloudflared tunnel create traffic-cnn

# Configurar ingress en ~/.cloudflared/config.yml
# Agregar el CNAME en Cloudflare DNS
cloudflared tunnel route dns <tunnel-id> tu-subdominio.tudominio.com
```

---

## Variables de entorno

No se requieren variables de entorno para el funcionamiento básico. El path del modelo está hardcodeado en `server/main.py`:

```python
model_path = "/home/uno21/models/baseline_cnn.pth"
```

Si cambias la ruta del modelo, actualiza esta línea antes de construir la imagen Docker.

---

## Solución de problemas comunes

**El contenedor no ve la GPU:**
```bash
docker exec traffic-api nvidia-smi
# Si falla, verifica que nvidia-container-toolkit esté instalado y Docker reiniciado
```

**Error 502 al acceder via Cloudflare:**
- Verifica que el contenedor de cloudflared tenga `extra_hosts: host.docker.internal:host-gateway`
- Verifica que el puerto 8000 esté abierto: `sudo firewall-cmd --add-port=8000/tcp --permanent`

**CUDA not available dentro del contenedor:**
```bash
# Verifica la versión de PyTorch instalada
docker exec traffic-api python -c "import torch; print(torch.version.cuda)"
# Debe coincidir con la versión de CUDA del driver del host
```

---

**Autor:** Erick Alberto Sánchez Aranda · A01641715  
**Institución:** Tecnológico de Monterrey, Guadalajara  
**Repositorio:** https://github.com/uno21858/traffic-sign-cnn
