#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de entrenamiento adaptado para Google Coral
Genera embeddings usando el modelo TFLite y entrena el clasificador SVM

Ejecutar en un entorno con TensorFlow y scikit-learn
(puede ser tu PC o Colab, luego transferir los archivos al Coral)
"""

import cv2 as cv
import os
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


class CoralFaceTraining:
    """
    Clase para entrenar el sistema de reconocimiento facial para Coral
    """
    
    def __init__(self, dataset_dir, use_keras=True):
        """
        Inicializa el sistema de entrenamiento
        
        Args:
            dataset_dir: Directorio con subdirectorios por persona
            use_keras: Si True, usa modelo Keras original (recomendado para entrenamiento)
        """
        self.dataset_dir = dataset_dir
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.use_keras = use_keras
        
        if use_keras:
            # Usar modelo Keras original (más confiable para entrenamiento)
            print("Cargando FaceNet (modelo Keras original)...")
            from keras_facenet import FaceNet
            self.embedder = FaceNet()
            print("✓ Modelo Keras cargado")
            print("  Input shape: (None, 160, 160, 3)")
            print("  Output shape: (None, 512)")
        else:
            # Usar modelo TFLite (para compatibilidad con Coral)
            tflite_model_path = 'models/facenet_embedding.tflite'
            print(f"Cargando modelo TFLite: {tflite_model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            self.interpreter.allocate_tensors()
            
            # Obtener detalles de entrada/salida
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✓ Modelo cargado")
            print(f"  Input shape: {self.input_details[0]['shape']}")
            print(f"  Output shape: {self.output_details[0]['shape']}")
    
    
    def detect_face_simple(self, img):
        """
        Detección simple usando Haar Cascade (para entrenamiento)
        En producción usaremos el modelo optimizado en Edge TPU
        """
        face_cascade = cv.CascadeClassifier(
            cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Tomar el primer rostro
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        
        return face
    
    
    def load_faces(self, person_dir):
        """
        Carga todas las imágenes de una persona
        
        Args:
            person_dir: Directorio con imágenes de una persona
            
        Returns:
            Lista de rostros extraídos
        """
        faces = []
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            # Ignorar archivos no válidos
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            try:
                # Leer imagen
                img = cv.imread(img_path)
                if img is None:
                    continue
                
                # Detectar rostro
                face = self.detect_face_simple(img)
                if face is None:
                    print(f"  ⚠ No se detectó rostro en: {img_name}")
                    continue
                
                # Convertir a RGB y redimensionar
                face_rgb = cv.cvtColor(face, cv.COLOR_BGR2RGB)
                face_resized = cv.resize(face_rgb, self.target_size)
                
                faces.append(face_resized)
                
            except Exception as e:
                print(f"  ✗ Error procesando {img_name}: {e}")
        
        return faces
    
    
    def load_dataset(self):
        """
        Carga todo el dataset
        
        Returns:
            (X, Y) - Arrays de rostros y etiquetas
        """
        print(f"\nCargando dataset desde: {self.dataset_dir}\n")
        
        for person_name in os.listdir(self.dataset_dir):
            person_dir = os.path.join(self.dataset_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
            
            print(f"Procesando: {person_name}")
            faces = self.load_faces(person_dir)
            
            if len(faces) == 0:
                print(f"  ⚠ No se encontraron rostros válidos\n")
                continue
            
            # Agregar rostros y etiquetas
            self.X.extend(faces)
            self.Y.extend([person_name] * len(faces))
            
            print(f"  ✓ Cargadas {len(faces)} imágenes\n")
        
        X = np.array(self.X)
        Y = np.array(self.Y)
        
        print(f"Total: {len(X)} imágenes de {len(np.unique(Y))} personas")
        
        return X, Y
    
    
    def get_embedding(self, face_img):
        """
        Obtiene el embedding usando Keras o TFLite
        
        Args:
            face_img: Imagen RGB de 160x160
            
        Returns:
            Vector de embedding
        """
        if self.use_keras:
            # Usar modelo Keras original
            face_normalized = face_img.astype('float32')
            face_input = np.expand_dims(face_normalized, axis=0)
            embedding = self.embedder.embeddings(face_input)
            return embedding[0]
        else:
            # Usar modelo TFLite
            input_type = self.input_details[0]['dtype']
            
            if input_type == np.uint8:
                # Modelo cuantizado - usar valores 0-255
                face_input = face_img.astype('uint8')
            else:
                # Modelo float - normalizar 0-1
                face_input = face_img.astype('float32') / 255.0
            
            face_input = np.expand_dims(face_input, axis=0)
            
            # Inferencia
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                face_input
            )
            self.interpreter.invoke()
            
            # Obtener output
            embedding = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            return embedding.flatten()
    
    
    def generate_embeddings(self, X):
        """
        Genera embeddings para todas las imágenes
        
        Args:
            X: Array de imágenes
            
        Returns:
            Array de embeddings
        """
        print("\nGenerando embeddings...")
        
        embeddings = []
        total = len(X)
        
        for i, face in enumerate(X):
            embedding = self.get_embedding(face)
            embeddings.append(embedding)
            
            if (i + 1) % 10 == 0:
                print(f"  Progreso: {i+1}/{total}")
        
        print(f"✓ {total} embeddings generados\n")
        
        return np.array(embeddings)
    
    
    def train_svm(self, X_embedded, Y):
        """
        Entrena el clasificador SVM
        
        Args:
            X_embedded: Embeddings
            Y: Etiquetas
            
        Returns:
            (model, encoder) - Modelo SVM y encoder de etiquetas
        """
        print("Entrenando clasificador SVM...")
        
        # Codificar etiquetas
        encoder = LabelEncoder()
        Y_encoded = encoder.fit_transform(Y)
        
        # Split train/test
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_embedded, Y_encoded, 
            test_size=0.2, 
            random_state=42,
            stratify=Y_encoded
        )
        
        # Entrenar SVM
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, Y_train)
        
        # Evaluar
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(Y_train, train_pred)
        test_acc = accuracy_score(Y_test, test_pred)
        
        print(f"\n✓ Entrenamiento completado")
        print(f"  Accuracy (train): {train_acc:.4f}")
        print(f"  Accuracy (test):  {test_acc:.4f}\n")
        
        # Reporte de clasificación
        print("Reporte de clasificación:")
        print(classification_report(
            Y_test, test_pred, 
            target_names=encoder.classes_
        ))
        
        return model, encoder
    
    
    def save_models(self, X_embedded, Y, model, encoder, 
                   embeddings_path='data/faces_embeddings.npz',
                   svm_path='models/svm_model_160x160.pkl'):
        """
        Guarda los modelos y datos
        """
        print("\nGuardando modelos...")
        
        # Guardar embeddings
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        np.savez_compressed(embeddings_path, X_embedded, Y)
        print(f"✓ Embeddings guardados: {embeddings_path}")
        
        # Guardar SVM
        os.makedirs(os.path.dirname(svm_path), exist_ok=True)
        with open(svm_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Modelo SVM guardado: {svm_path}")
        
        print("\n✓ Proceso completado")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Entrenamiento de reconocimiento facial para Coral'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Directorio del dataset (con subdirectorios por persona)'
    )
    parser.add_argument(
        '--use-tflite',
        action='store_true',
        help='Usar modelo TFLite en lugar de Keras (puede tener problemas)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Entrenamiento para Google Coral")
    print("=" * 60)
    
    # Inicializar (por defecto usa Keras, más estable)
    trainer = CoralFaceTraining(args.dataset, use_keras=not args.use_tflite)
    
    # Cargar dataset
    X, Y = trainer.load_dataset()
    
    if len(X) == 0:
        print("✗ Error: No se cargaron imágenes")
        return
    
    # Generar embeddings
    X_embedded = trainer.generate_embeddings(X)
    
    # Entrenar SVM
    model, encoder = trainer.train_svm(X_embedded, Y)
    
    # Guardar
    trainer.save_models(X_embedded, Y, model, encoder)
    
    print("\n" + "=" * 60)
    print("Próximos pasos:")
    print("1. Copia models/svm_model_160x160.pkl a tu Coral")
    print("2. Copia data/faces_embeddings.npz a tu Coral")
    print("3. Ejecuta coral_main.py en tu Coral Dev Board")
    print("=" * 60)


if __name__ == "__main__":
    main()
