from dlib_facial_landmark import *
from image_processing import *

def apply_hat(frame, shape, face):
    """
    Apply a hat filter to the detected face in the frame.

    Args:
        frame (numpy.ndarray): The image frame.
        shape (numpy.ndarray): Facial landmark points.
        face (dlib.rectangle): Detected face object.

    Returns:
        None
    """
    hat_x, hat_y, hat_w, hat_h = (face.left(), face.top(), face.width(), face.height())
    use_filters(frame, "./filters/hat4.png", hat_w, hat_x, hat_y, compute_angle_between_coords(shape[17], shape[26]))


def apply_mustache(frame, shape):
    """
    Apply a mustache filter to the detected face in the frame.

    Args:
        frame (numpy.ndarray): The image frame.
        shape (numpy.ndarray): Facial landmark points.

    Returns:
        None
    """
    m_x, m_y, m_w, m_h = determine_feature_boundary(shape, 6)
    m_y += 35
    use_filters(frame, "./filters/mustache3.png", m_w, m_x, m_y, compute_angle_between_coords(shape[17], shape[26]))


def apply_glasses(frame, shape, face):
    """
    Apply glasses filter to the detected face in the frame.

    Args:
        frame (numpy.ndarray): The image frame.
        shape (numpy.ndarray): Facial landmark points.
        face (dlib.rectangle): Detected face object.

    Returns:
        None
    """

    g_x, g_y, g_w, g_h = determine_feature_boundary(shape, 1)
    use_filters(frame, "./filters/glasses.png", face.width(), face.left(), g_y,
                compute_angle_between_coords(shape[17], shape[26]), overlay=False)


def apply_beard(frame, shape):
    """
    Apply a beard filter to the detected face in the frame.

    Args:
        frame (numpy.ndarray): The image frame.
        shape (numpy.ndarray): Facial landmark points.

    Returns:
        None
    """
    b_x, b_y, b_w, b_h = determine_feature_boundary(shape, 7)
    use_filters(frame, "./filters/beard.png", int(b_w * 4), b_x - 155, b_y + 250,
                compute_angle_between_coords(shape[17], shape[26]))