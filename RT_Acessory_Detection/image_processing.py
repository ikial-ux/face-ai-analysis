import cv2
from imutils import rotate_bound


def toggle_filter(value, my_filters):
    """
    Toggle the state of a filter at the given index in the FILTERS list.

    Args:
        value (int): Index of the filter to toggle.
        my_filters (list): List representing the state of filters.


    Returns:
        list: Updated FILTERS list after toggling the specified filter.
    """

    my_filters[value] = 1 - my_filters[value]
    return my_filters


def apply_overlay(base_img, overlay_img, x, y):
    """
    Apply an overlay image onto the base image at the specified position offsets.

    Args:
        base_img (numpy.ndarray): Base image.
        overlay_img (numpy.ndarray): Overlay image.
        x (int): X-axis position offset for the overlay.
        y (int): Y-axis position offset for the overlay.

    Returns:
        numpy.ndarray: Image with the overlay applied.
    """
    overlay_h, overlay_w = overlay_img.shape[0], overlay_img.shape[1]
    base_h, base_w = base_img.shape[0], base_img.shape[1]

    if y + overlay_h >= base_h:
        overlay_img = overlay_img[0: base_h - y, :, :]

    if x + overlay_w >= base_w:
        overlay_img = overlay_img[:, 0: base_w - x, :]

    if x < 0:
        overlay_w = overlay_img[:, abs(x)::, :].shape[1]
        x = 0

    for color_ch in range(3):
        base_img[y: y + overlay_h, x: x + overlay_w, color_ch] = (
                    overlay_img[:, :, color_ch] * (overlay_img[:, :, 3] / 255.0)
                    + base_img[y: y + overlay_h, x: x + overlay_w, color_ch] *
                    (1.0 - overlay_img[:, :, 3] / 255.0))
    return base_img


def calibrate_filter(input_image, head_size, head_pos_y, overlay=True):
    """
    Modify the input image's size and position to fit the head region.

    Args:
        input_image (numpy.ndarray): Input image.
        head_size (int): Width of the head.
        head_pos_y (int): Y-axis position of the head.
        overlay (bool, optional): Flag to overlay the modified image on the head. Defaults to True.

    Returns:
        numpy.ndarray: Adjusted image.
        int: Y-axis position of the adjusted image.
    """
    base_h, base_w = input_image.shape[0], input_image.shape[1]
    scaling_factor = 1.0 * head_size / base_w
    input_image = cv2.resize(input_image, (0, 0), fx=scaling_factor, fy=scaling_factor)
    base_h, base_w = input_image.shape[0], input_image.shape[1]

    y_position = head_pos_y - base_h if overlay else head_pos_y
    if y_position < 0:
        input_image = input_image[abs(y_position)::, :, :]
        y_position = 0

    return input_image, y_position


def use_filters(image, filter_dir, w, x, y, angle, overlay=True):
    """
    Apply a sprite to the image at the specified coordinates and adjust it to the head's position.

    Args:
        image (numpy.ndarray): Image to which the sprite will be applied.
        filter_dir (str): Path to the sprite image file.
        w (int): Width of the sprite.
        x (int): X-axis position for placing the sprite.
        y (int): Y-axis position for placing the sprite.
        angle (float): Angle for rotating the sprite.
        overlay (bool, optional): Flag to place the sprite on top of the head. Defaults to True.

    Returns:
        numpy.ndarray: Image with the sprite applied.
    """
    overlay_img = rotate_bound(cv2.imread(filter_dir, -1), angle)
    overlay_img, y = calibrate_filter(overlay_img, w, y, overlay)
    apply_overlay(image, overlay_img, x, y)
