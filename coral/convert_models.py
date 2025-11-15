#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para convertir modelos MTCNN y FaceNet a TensorFlow Lite
con cuantización INT8 para Google Coral Edge TPU

Ejecutar en un entorno con TensorFlow instalado (tu PC, no en Coral)
"""

import tensorflow as tf
import numpy as np
from keras_facenet import FaceNet
import os

def convert_facenet_to_tflite():
    """
    Convierte FaceNet a TFLite (Float32 optimizado, compatible con entrenamiento)
    """
    print("Cargando FaceNet...")
    embedder = FaceNet()
    model = embedder.model
    
    # Verificar input shape
    print(f"Modelo original - Input shape: {model.input_shape}")
    print(f"Modelo original - Output shape: {model.output_shape}")
    
    print("\nConvirtiendo FaceNet a TFLite (Float32 optimizado)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimización dinámica (mantiene float pero reduce tamaño)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # NO forzar INT8 en entrada/salida (causa problemas de shape)
    # Esto genera un modelo float32 optimizado
    
    tflite_model = converter.convert()
    
    # Guardar modelo
    output_path = 'models/facenet_embedding.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"\n✓ Modelo Float32 guardado en: {output_path}")
    print(f"  Tamaño: {len(tflite_model) / 1024:.2f} KB")
    
    # Verificar modelo convertido
    print("\nVerificando modelo convertido...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input type: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output type: {output_details[0]['dtype']}")
    
    if list(input_details[0]['shape']) != [1, 160, 160, 3]:
        print("\n⚠ ADVERTENCIA: El modelo no tiene el shape esperado [1, 160, 160, 3]")
        print("  Esto puede causar problemas durante el entrenamiento.")
    
    return output_path


def download_face_detection_model():
    """
    Descarga modelo SSD MobileNet para detección de rostros optimizado para Edge TPU
    """
    print("\n=== Descargando modelo de detección de rostros ===")
    print("Nota: Este modelo ya está optimizado para Edge TPU")
    
    import urllib.request
    
    models = {
        'face_detection_edgetpu.tflite': 
            'https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite',
        'face_detection.tflite':
            'https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_face_quant_postprocess.tflite'
    }
    
    for filename, url in models.items():
        output_path = f'models/{filename}'
        if os.path.exists(output_path):
            print(f"✓ {filename} ya existe")
            continue
        
        print(f"Descargando {filename}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            size = os.path.getsize(output_path) / 1024
            print(f"✓ Descargado: {output_path} ({size:.2f} KB)")
        except Exception as e:
            print(f"✗ Error descargando {filename}: {e}")


def compile_for_edgetpu(tflite_model_path):
    """
    Instrucciones para compilar el modelo para Edge TPU
    (Requiere Edge TPU Compiler instalado)
    """
    print("\n=== Compilación para Edge TPU ===")
    print("Para compilar el modelo FaceNet para Edge TPU, ejecuta:")
    print(f"\n  edgetpu_compiler {tflite_model_path}\n")
    print("Esto generará: facenet_embedding_edgetpu.tflite")
    print("\nInstalación del compilador:")
    print("  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -")
    print("  echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main' | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list")
    print("  sudo apt-get update")
    print("  sudo apt-get install edgetpu-compiler")


def main():
    print("=" * 60)
    print("Conversión de modelos para Google Coral Edge TPU")
    print("=" * 60)
    
    # Crear directorio si no existe
    os.makedirs('models', exist_ok=True)
    
    # 1. Convertir FaceNet
    facenet_path = convert_facenet_to_tflite()
    
    # 2. Descargar modelo de detección
    download_face_detection_model()
    
    # 3. Instrucciones para compilar
    compile_for_edgetpu(facenet_path)
    
    print("\n" + "=" * 60)
    print("✓ Proceso completado")
    print("=" * 60)
    print("\nPróximos pasos:")
    print("1. Compila facenet_embedding.tflite para Edge TPU (ver instrucciones arriba)")
    print("2. Copia los archivos .tflite a tu Coral Dev Board")
    print("3. Copia también tu archivo svm_model_160x160.pkl")
    print("4. Ejecuta el script coral_main.py en tu Coral")


if __name__ == "__main__":
    main()
