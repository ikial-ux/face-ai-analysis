from djitellopy import Tello
from deepface import DeepFace
import cv2
import mediapipe as mp
import time

tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

tello.streamon()
frame_reader = tello.get_frame_read()

# tello.takeoff()
time.sleep(2)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.2)

frame_count = 0
info = None

try:
    while True:
        frame = frame_reader.frame
        if frame is None:
            continue

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = face_detection.process(rgb)

        if detections.detections:
            h, w, _ = frame.shape
            for det in detections.detections:
                box = det.location_data.relative_bounding_box
                x1 = int(box.xmin * w)
                y1 = int(box.ymin * h)
                x2 = x1 + int(box.width * w)
                y2 = y1 + int(box.height * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            if frame_count % 120 == 0:
                # scale down for faster analysis
                small = cv2.resize(rgb, (0,0), fx=0.5, fy=0.5)
                try:
                    result = DeepFace.analyze(
                        small,
                        actions=['age','gender','race'],
                        enforce_detection=False
                    )
                    if isinstance(result, list):
                        result = result[0]
                    info = result
                except Exception as e:
                    print("DeepFace error:", e)
                    info = None

        # 6) Mostrar info sobre el frame
        if info:
            age = info.get('age','?')
            race = info.get('dominant_race','?').lower()
            if race in ['middle eastern','middle-east']:
                race = 'jewish'
            gender = info.get('gender')
            if isinstance(gender, dict):
                # pick the most probable
                gender = max(gender, key=gender.get)

            cv2.putText(frame, f"Gender: {gender}", (20,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Age: {age}",    (20,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Race: {race}", (20,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # 7) Mostrar ventana
        cv2.imshow("Tello Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

        # (Opcional) mantener vivo el dron
        if frame_count % 100 == 0:
            tello.send_rc_control(0,0,0,0)  # no-movement, solo keep-alive

finally:
    # Cleanup
    cv2.destroyAllWindows()
    tello.streamoff()
    tello.end()