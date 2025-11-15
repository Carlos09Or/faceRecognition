# Reconocimiento Facial para Google Coral Dev Board 5.3

Sistema completo de reconocimiento facial optimizado para Google Coral Edge TPU, adaptado desde el proyecto original que usa MTCNN + FaceNet + SVM.

## 🎯 Diferencias principales con el proyecto original

| Componente | PC (Original) | Coral (Optimizado) |
|-----------|---------------|-------------------|
| **Detección** | MTCNN (TensorFlow) | SSD MobileNet V2 (TFLite + Edge TPU) |
| **Embeddings** | FaceNet (Keras) | FaceNet cuantizado (TFLite + Edge TPU) |
| **Clasificación** | SVM (CPU) | SVM (CPU) |
| **Framework** | TensorFlow/Keras | TensorFlow Lite + PyCoral |

---

## 📁 Estructura del proyecto

```
coral/
├── convert_models.py           # Conversión de modelos a TFLite
├── train_coral.py              # Entrenamiento con modelos TFLite
├── coral_face_recognition.py   # Clase principal de reconocimiento
├── coral_main.py               # Aplicación principal
├── README.md                   # Este archivo
├── models/                     # Modelos optimizados
│   ├── face_detection_edgetpu.tflite
│   ├── facenet_embedding_edgetpu.tflite
│   └── svm_model_160x160.pkl
└── data/                       # Datos de entrenamiento
    └── faces_embeddings.npz
```

---

## 🚀 Guía de uso completa

### **Paso 1: Preparación en tu PC**

#### 1.1. Instalar dependencias
```bash
pip install tensorflow keras-facenet opencv-python scikit-learn numpy
```

#### 1.2. Convertir modelos a TFLite
```bash
cd coral
python convert_models.py
```

Esto generará:
- `models/facenet_embedding.tflite` - Modelo de embeddings
- `models/face_detection_edgetpu.tflite` - Detector de rostros (descargado)

#### 1.3. Compilar para Edge TPU

**En Linux o WSL:**
```bash
# Instalar Edge TPU Compiler
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler

# Compilar modelo
edgetpu_compiler models/facenet_embedding.tflite
```

Esto genera: `facenet_embedding_edgetpu.tflite`

#### 1.4. Entrenar el clasificador SVM
```bash
python train_coral.py --dataset /path/to/dataset --model models/facenet_embedding.tflite
```

**Estructura del dataset:**
```
dataset/
├── Persona1/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── ...
├── Persona2/
│   ├── foto1.jpg
│   └── ...
└── ...
```

Esto generará:
- `models/svm_model_160x160.pkl`
- `data/faces_embeddings.npz`

---

### **Paso 2: Configuración del Coral Dev Board**

#### 2.1. Conectar al Coral

**Por SSH:**
```bash
ssh mendel@coral-dev-board.local
# Contraseña por defecto: mendel
```

**Por puerto serial:**
```bash
screen /dev/ttyUSB0 115200
```

#### 2.2. Instalar dependencias en Coral

```bash
# Actualizar sistema
sudo apt-get update
sudo apt-get upgrade

# Instalar Python y pip
sudo apt-get install python3-pip python3-opencv

# Instalar PyCoral
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-pycoral

# Instalar otras dependencias
pip3 install scikit-learn numpy opencv-python
```

#### 2.3. Transferir archivos al Coral

**Desde tu PC:**
```bash
# Crear directorio en Coral
ssh mendel@coral-dev-board.local "mkdir -p ~/face_recognition/coral/{models,data}"

# Copiar scripts
scp coral_*.py mendel@coral-dev-board.local:~/face_recognition/coral/

# Copiar modelos
scp models/face_detection_edgetpu.tflite mendel@coral-dev-board.local:~/face_recognition/coral/models/
scp models/facenet_embedding_edgetpu.tflite mendel@coral-dev-board.local:~/face_recognition/coral/models/
scp models/svm_model_160x160.pkl mendel@coral-dev-board.local:~/face_recognition/coral/models/

# Copiar datos
scp data/faces_embeddings.npz mendel@coral-dev-board.local:~/face_recognition/coral/data/
```

---

### **Paso 3: Ejecutar en Coral**

#### 3.1. Modo con display

```bash
ssh mendel@coral-dev-board.local
cd ~/face_recognition/coral
python3 coral_main.py
```

#### 3.2. Modo sin display (ejecución remota)

```bash
python3 coral_main.py --no-display
```

#### 3.3. Opciones avanzadas

```bash
python3 coral_main.py \
  --camera 0 \
  --width 640 \
  --height 480 \
  --confidence 0.5 \
  --recognition-threshold 0.6 \
  --detection-model models/face_detection_edgetpu.tflite \
  --embedding-model models/facenet_embedding_edgetpu.tflite \
  --svm-model models/svm_model_160x160.pkl \
  --embeddings-db data/faces_embeddings.npz
```

**Parámetros:**
- `--camera`: ID de la cámara (default: 0)
- `--confidence`: Umbral de detección (0.0-1.0)
- `--recognition-threshold`: Umbral de reconocimiento (0.0-1.0)
- `--no-display`: Sin ventana (para SSH)

---

## 🔧 Troubleshooting

### Error: "No module named 'pycoral'"
```bash
sudo apt-get install python3-pycoral
```

### Error: "Failed to load delegate from edgetpu.so"
```bash
# Reinstalar runtime de Edge TPU
sudo apt-get install --reinstall libedgetpu1-std
```

### Cámara no detectada
```bash
# Listar cámaras disponibles
v4l2-ctl --list-devices

# Probar con otro ID
python3 coral_main.py --camera 1
```

### Bajo FPS
- Reducir resolución: `--width 320 --height 240`
- Verificar que los modelos tengan sufijo `_edgetpu.tflite`
- Asegurar que Edge TPU esté habilitado

### Error de permisos
```bash
# Agregar usuario al grupo necesario
sudo usermod -aG video mendel
sudo reboot
```

---

## 📊 Rendimiento esperado

| Métrica | Valor esperado |
|---------|----------------|
| **FPS** | 15-25 fps @ 640x480 |
| **Latencia de inferencia** | 30-50 ms |
| **Detección** | ~10 ms (Edge TPU) |
| **Embedding** | ~20 ms (Edge TPU) |
| **SVM** | ~1 ms (CPU) |

---

## 🔄 Actualización del modelo

Si entrenas un nuevo modelo SVM:

```bash
# 1. En tu PC: entrenar
python train_coral.py --dataset /path/to/new/dataset

# 2. Copiar al Coral
scp models/svm_model_160x160.pkl mendel@coral-dev-board.local:~/face_recognition/coral/models/
scp data/faces_embeddings.npz mendel@coral-dev-board.local:~/face_recognition/coral/data/

# 3. Reiniciar aplicación en Coral
```

---

## 📝 Controles durante ejecución

| Tecla | Acción |
|-------|--------|
| `q` | Salir |
| `s` | Capturar pantalla |

---

## 🔗 Referencias

- [Google Coral Documentation](https://coral.ai/docs/)
- [PyCoral API](https://coral.ai/docs/reference/pycoral/)
- [Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)

---

## 📧 Soporte

Si encuentras problemas:
1. Verifica que Edge TPU esté correctamente instalado: `lsusb` (debe aparecer "Google Inc.")
2. Revisa los logs del sistema: `dmesg | grep apex`
3. Prueba los ejemplos oficiales de Coral: `python3 /usr/share/edgetpudemo`

---

## ✅ Checklist de deployment

- [ ] Modelos convertidos a TFLite
- [ ] Modelo compilado para Edge TPU (`_edgetpu.tflite`)
- [ ] SVM entrenado con dataset
- [ ] PyCoral instalado en Coral
- [ ] Archivos transferidos al Coral
- [ ] Cámara conectada y funcionando
- [ ] Permisos de acceso a dispositivos configurados
- [ ] Aplicación ejecutándose correctamente

---

**¡Tu sistema de reconocimiento facial está listo para Google Coral! 🎉**
