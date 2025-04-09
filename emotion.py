import cv2
import sys
import os
from deepface import DeepFace

# Carreguem el classificador de cares preentrenat
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciem la captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Llegim un fotograma de la càmera
    ret, frame = cap.read()
    if not ret:
        break

    # Convertim el fotograma a escala de grisos
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectem cares en el fotograma
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extraïm la regió de la cara detectada
        face_roi = frame[y:y + h, x:x + w]

        try:
            # Utilitzem DeepFace per analitzar l'emoció amb el model predefinit (VGG-Face)
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Obtenim l'emoció predominant
            emotion = analysis[0]['dominant_emotion']

            # Dibuixem un rectangle al voltant de la cara i afegim el text de l'emoció detectada
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error en l'anàlisi: {e}")

    # Mostrem el fotograma amb les emocions detectades
    cv2.imshow('Real-time Emotion Detection', frame)

    # Capturem la tecla premuda
    key = cv2.waitKey(1) & 0xFF

    # Si es prem 'q' o es tanca la finestra, sortim del bucle
    if key == ord('q'):
        break
    if cv2.getWindowProperty('Real-time Emotion Detection', cv2.WND_PROP_AUTOSIZE) < 0:
        break
    
# Alliberem la càmera i tanquem totes les finestres
cap.release()
cv2.destroyAllWindows()

# Forcem el tancament immediat del procés
os._exit(0)
os.kill(os.getpid(), signal.SIGTERM)