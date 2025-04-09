import argparse
import threading
import time
from threading import Thread
import dlib
from gui_handling import *
from imutils import face_utils
from apply_filters import *

def initialize_webcam_models(read_camera):
    """
    Initializes the webcam models including video capture, face detector, and predictor.

    Args:
        read_camera (int): ID to read camera from.

    Returns:
        tuple: Video capture object, face detector, and shape predictor objects.
    """
    model = "models/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(model)

    return cv2.VideoCapture(read_camera), dlib.get_frontal_face_detector(), predictor


def webcam_processing(run_event_flag, camera_id=0, virtual_cam_id=0):
    """
    Continuous loop for handling webcam frames and applying filters.

    Args:
        run_event_flag (threading.Event): Event to control the loop.
        camera_id (int, optional): Camera ID to read from. Defaults to 0.
        virtual_cam_id (int, optional): ID for virtual camera output (Linux only). Defaults to 0.

    Returns:
        None
    """
    cap, detector, predictor = initialize_webcam_models(camera_id)

    while run_event_flag.is_set():
        ret, frame = cap.read()

        if not ret:
            print("Error reading camera, exiting")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = detector(gray_frame, 0)

        for detected_face in detected_faces:
            shape = predictor(gray_frame, detected_face)
            shape = face_utils.shape_to_np(shape)

            if my_filters[0]:
                apply_hat(frame, shape, detected_face)

            if my_filters[1]:
                apply_mustache(frame, shape)

            if my_filters[2]:
                apply_beard(frame, shape)

            if my_filters[3]:
                apply_glasses(frame, shape, detected_face)

            display_updated_image(frame, updated_panel)

    cap.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--read_camera", type=int, default=0, help="Id to read camera from")
    parser.add_argument(
        "--virtual_camera",
        type=int,
        default=0,
        help="If different from 0, creates a virtual camera with results on that id (linux only)",
    )
    args = parser.parse_args()

    my_filters, gui, updated_panel = setup_gui()

    run_event = threading.Event()
    run_event.set()
    action = Thread(target=webcam_processing, args=(run_event, args.read_camera, args.virtual_camera))
    action.daemon = True
    action.start()

    gui.protocol("WM_DELETE_WINDOW", lambda: [run_event.clear(), time.sleep(1), gui.destroy(), print("Closing Project!")])
    gui.mainloop()
