from deepface import DeepFace
import cv2
import mediapipe as mp

# Mediapipe setup
detros = mp.solutions.face_detection
face_detection = detros.FaceDetection(min_detection_confidence=0.2)

cap = cv2.VideoCapture(1)
frame_count = 0
info = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faceres = face_detection.process(rgb)

    if faceres.detections:
        for face in faceres.detections:
            al, an, _ = frame.shape
            box = face.location_data.relative_bounding_box
            xi = int(box.xmin * an)
            yi = int(box.ymin * al)
            w = int(box.width * an)
            h = int(box.height * al)
            x, y = xi + w, yi + h

            cv2.rectangle(frame, (xi, yi), (x, y), (255, 255, 0), 1)

            # Analizar solo cada 15 frames para mejorar rendimiento
            if frame_count % 120 == 0:
                small_rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)

                try:
                    result = DeepFace.analyze(
                        small_rgb,
                        actions=['age', 'gender', 'race'],
                        enforce_detection=False
                    )
                    if isinstance(result, list):
                        result = result[0]
                    info = result
                except Exception as e:
                    print("Error en DeepFace:", e)
                    info = None

            if info:
                age = info.get('age', '?')
                race = info.get('dominant_race', '?')
                if race.lower() in ['middle eastern', 'middle-east']:
                    race = 'jewish'
                gender_dict = info.get('gender', {})
                gen = max(gender_dict, key=gender_dict.get) if isinstance(gender_dict, dict) else gender_dict

                cv2.putText(frame, f"Genero: {gen}", (65, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Edad: {age}", (65, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Raza: {race}", (65, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()