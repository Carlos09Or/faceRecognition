#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicación principal de reconocimiento facial para Google Coral Dev Board
Ejecuta reconocimiento facial en tiempo real usando la cámara
"""

import cv2 as cv
import time
import argparse
from coral_face_recognition import CoralFaceRecognition


def draw_results(frame, results):
    """
    Dibuja los resultados del reconocimiento en el frame
    
    Args:
        frame: Imagen BGR de OpenCV
        results: Lista de diccionarios con detecciones
    """
    for result in results:
        x1, y1, x2, y2 = result['bbox']
        label = result['label']
        confidence = result['confidence']
        
        # Color según reconocimiento
        if label == "Desconocido":
            color = (0, 0, 255)  # Rojo
        else:
            color = (0, 255, 0)  # Verde
        
        # Dibujar rectángulo
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Preparar texto
        text = f"{label} ({confidence:.2f})"
        
        # Fondo para el texto
        (text_width, text_height), _ = cv.getTextSize(
            text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv.rectangle(
            frame, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # Dibujar texto
        cv.putText(
            frame, 
            text, 
            (x1, y1 - 5), 
            cv.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
    
    return frame


def main():
    parser = argparse.ArgumentParser(
        description='Reconocimiento Facial en Google Coral Dev Board'
    )
    parser.add_argument(
        '--detection-model',
        default='models/face_detection_edgetpu.tflite',
        help='Ruta al modelo de detección TFLite'
    )
    parser.add_argument(
        '--embedding-model',
        default='models/facenet_embedding_edgetpu.tflite',
        help='Ruta al modelo de embeddings TFLite'
    )
    parser.add_argument(
        '--svm-model',
        default='models/svm_model_160x160.pkl',
        help='Ruta al clasificador SVM'
    )
    parser.add_argument(
        '--embeddings-db',
        default='data/faces_embeddings.npz',
        help='Ruta a la base de datos de embeddings'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='ID de la cámara (default: 0)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Umbral de confianza para detección (0.0-1.0)'
    )
    parser.add_argument(
        '--recognition-threshold',
        type=float,
        default=0.6,
        help='Umbral de confianza para reconocimiento (0.0-1.0)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Ancho del video (default: 640)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Alto del video (default: 480)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='No mostrar ventana (útil para ejecución remota)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Reconocimiento Facial - Google Coral Dev Board")
    print("=" * 60)
    
    # Inicializar sistema de reconocimiento
    try:
        recognizer = CoralFaceRecognition(
            detection_model_path=args.detection_model,
            embedding_model_path=args.embedding_model,
            svm_model_path=args.svm_model,
            embeddings_db_path=args.embeddings_db,
            confidence_threshold=args.confidence,
            recognition_threshold=args.recognition_threshold
        )
    except Exception as e:
        print(f"✗ Error al inicializar el sistema: {e}")
        return
    
    # Inicializar cámara
    print(f"Abriendo cámara {args.camera}...")
    cap = cv.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("✗ Error: No se pudo abrir la cámara")
        return
    
    # Configurar resolución
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    print("✓ Cámara iniciada correctamente")
    print("\nControles:")
    print("  'q' - Salir")
    print("  's' - Tomar captura")
    print("=" * 60)
    print()
    
    # Variables para FPS
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    try:
        while True:
            # Capturar frame
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Error al capturar frame")
                break
            
            # Procesar frame
            start_time = time.time()
            results = recognizer.process_frame(frame)
            inference_time = (time.time() - start_time) * 1000
            
            # Dibujar resultados
            frame = draw_results(frame, results)
            
            # Calcular FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Dibujar información de rendimiento
            info_text = f"FPS: {fps} | Inferencia: {inference_time:.1f}ms | Rostros: {len(results)}"
            cv.putText(
                frame,
                info_text,
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Mostrar frame
            if not args.no_display:
                cv.imshow('Coral Face Recognition', frame)
            
            # Procesar teclas
            key = cv.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nSaliendo...")
                break
            elif key == ord('s'):
                filename = f'capture_{int(time.time())}.jpg'
                cv.imwrite(filename, frame)
                print(f"✓ Captura guardada: {filename}")
    
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario")
    
    except Exception as e:
        print(f"\n✗ Error durante la ejecución: {e}")
    
    finally:
        # Liberar recursos
        cap.release()
        cv.destroyAllWindows()
        print("\n✓ Recursos liberados")


if __name__ == "__main__":
    main()
