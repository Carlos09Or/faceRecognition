#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Reconocimiento Facial Optimizado para Google Coral Dev Board
Utiliza modelos TFLite cuantizados ejecutados en Edge TPU
"""

import cv2 as cv
import numpy as np
import pickle
from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.detect import BBox
from sklearn.preprocessing import LabelEncoder


class CoralFaceRecognition:
    """
    Sistema de reconocimiento facial optimizado para Coral Edge TPU
    
    Componentes:
    - Detección de rostros: SSD MobileNet V2 en Edge TPU
    - Embeddings faciales: FaceNet cuantizado en Edge TPU
    - Clasificación: SVM en CPU
    """
    
    def __init__(self, 
                 detection_model_path='models/face_detection_edgetpu.tflite',
                 embedding_model_path='models/facenet_embedding_edgetpu.tflite',
                 svm_model_path='models/svm_model_160x160.pkl',
                 embeddings_db_path='data/faces_embeddings.npz',
                 confidence_threshold=0.5,
                 recognition_threshold=0.6):
        """
        Inicializa el sistema de reconocimiento facial
        
        Args:
            detection_model_path: Ruta al modelo de detección TFLite
            embedding_model_path: Ruta al modelo de embeddings TFLite
            svm_model_path: Ruta al clasificador SVM
            embeddings_db_path: Ruta a la base de datos de embeddings
            confidence_threshold: Umbral para detección (0.0-1.0)
            recognition_threshold: Umbral para reconocimiento (0.0-1.0)
        """
        print("Inicializando Coral Face Recognition...")
        
        # Cargar modelo de detección en Edge TPU
        print(f"Cargando detector: {detection_model_path}")
        self.face_detector = make_interpreter(detection_model_path)
        self.face_detector.allocate_tensors()
        
        # Cargar modelo de embeddings en Edge TPU
        print(f"Cargando modelo de embeddings: {embedding_model_path}")
        self.embedding_model = make_interpreter(embedding_model_path)
        self.embedding_model.allocate_tensors()
        
        # Cargar clasificador SVM
        print(f"Cargando clasificador SVM: {svm_model_path}")
        with open(svm_model_path, 'rb') as f:
            self.svm_model = pickle.load(f)
        
        # Cargar base de datos de embeddings (opcional)
        try:
            print(f"Cargando base de datos: {embeddings_db_path}")
            data = np.load(embeddings_db_path)
            self.known_embeddings = data['arr_0']
            self.known_labels = data['arr_1']
            
            # Crear encoder de etiquetas
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.known_labels)
        except:
            print("Advertencia: No se pudo cargar la base de datos de embeddings")
            self.known_embeddings = None
            self.known_labels = None
            self.label_encoder = None
        
        self.target_size = (160, 160)
        self.confidence_threshold = confidence_threshold
        self.recognition_threshold = recognition_threshold
        
        print("✓ Sistema inicializado correctamente\n")
    
    
    def detect_faces(self, frame):
        """
        Detecta rostros en un frame usando Edge TPU
        
        Args:
            frame: Imagen BGR de OpenCV
            
        Returns:
            Lista de detecciones (BBox objects)
        """
        # Convertir BGR a RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Redimensionar y preparar entrada
        common.set_input(self.face_detector, rgb_frame)
        
        # Ejecutar inferencia en Edge TPU
        self.face_detector.invoke()
        
        # Obtener detecciones
        faces = detect.get_objects(
            self.face_detector, 
            score_threshold=self.confidence_threshold
        )
        
        return faces
    
    
    def extract_face(self, frame, bbox):
        """
        Extrae y preprocesa un rostro detectado
        
        Args:
            frame: Imagen BGR original
            bbox: BBox object de la detección
            
        Returns:
            Rostro redimensionado a 160x160
        """
        height, width = frame.shape[:2]
        
        # Convertir coordenadas normalizadas a píxeles
        x1 = int(bbox.xmin * width)
        y1 = int(bbox.ymin * height)
        x2 = int(bbox.xmax * width)
        y2 = int(bbox.ymax * height)
        
        # Asegurar coordenadas dentro de límites
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        # Extraer rostro
        face = frame[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        # Convertir a RGB y redimensionar
        face_rgb = cv.cvtColor(face, cv.COLOR_BGR2RGB)
        face_resized = cv.resize(face_rgb, self.target_size)
        
        return face_resized
    
    
    def get_embedding(self, face_img):
        """
        Obtiene el embedding facial usando Edge TPU
        
        Args:
            face_img: Imagen RGB de 160x160
            
        Returns:
            Vector de embedding (512 dimensiones)
        """
        # Normalizar imagen
        face_normalized = face_img.astype('float32') / 255.0
        
        # Preparar entrada para el modelo
        common.set_input(self.embedding_model, face_normalized)
        
        # Ejecutar inferencia en Edge TPU
        self.embedding_model.invoke()
        
        # Obtener embedding
        embedding = common.output_tensor(self.embedding_model, 0)
        
        return embedding.flatten()
    
    
    def recognize_face(self, embedding):
        """
        Reconoce una persona a partir de su embedding
        
        Args:
            embedding: Vector de embedding facial
            
        Returns:
            (nombre, confianza) o ('Desconocido', 0.0)
        """
        try:
            # Predecir con SVM
            prediction = self.svm_model.predict([embedding])[0]
            probabilities = self.svm_model.predict_proba([embedding])[0]
            confidence = probabilities.max()
            
            # Verificar umbral de confianza
            if confidence >= self.recognition_threshold:
                # Decodificar etiqueta
                if self.label_encoder:
                    label = self.label_encoder.inverse_transform([prediction])[0]
                else:
                    label = str(prediction)
                return label, confidence
            else:
                return "Desconocido", confidence
                
        except Exception as e:
            print(f"Error en reconocimiento: {e}")
            return "Error", 0.0
    
    
    def process_frame(self, frame):
        """
        Procesa un frame completo: detecta rostros y los reconoce
        
        Args:
            frame: Imagen BGR de OpenCV
            
        Returns:
            Lista de diccionarios con información de cada rostro:
            [{'bbox': (x1,y1,x2,y2), 'label': 'Nombre', 'confidence': 0.95}, ...]
        """
        results = []
        
        # Detectar rostros
        faces = self.detect_faces(frame)
        
        height, width = frame.shape[:2]
        
        # Procesar cada rostro detectado
        for face_bbox in faces:
            # Extraer rostro
            face_img = self.extract_face(frame, face_bbox)
            
            if face_img is None:
                continue
            
            # Obtener embedding
            embedding = self.get_embedding(face_img)
            
            # Reconocer persona
            label, confidence = self.recognize_face(embedding)
            
            # Convertir bbox a coordenadas de píxeles
            x1 = int(face_bbox.xmin * width)
            y1 = int(face_bbox.ymin * height)
            x2 = int(face_bbox.xmax * width)
            y2 = int(face_bbox.ymax * height)
            
            results.append({
                'bbox': (x1, y1, x2, y2),
                'label': label,
                'confidence': confidence
            })
        
        return results
