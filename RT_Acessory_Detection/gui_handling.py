from image_processing import *
from PIL import Image, ImageTk
from tkinter import Tk, Label, ttk


def setup_gui():
    """
    Set up the graphical user interface.

    Returns:
        tuple: my_filters list, gui Tk object, buttons, and panel for image display.
    """
    gui = Tk()
    gui.title("Computer Vision Project")
    gui.config(bg="yellow")
    style = ttk.Style()
    style.configure('My.TButton', background='red', foreground='white')
    my_filters = [0, 0, 0, 0, 0]

    button_info = [
        ('Beard', 2),
        ('Mustache', 1),
        ('Glasses', 3),
        ('Hat', 0)
    ]

    for text, filter_index in button_info:
        btn = ttk.Button(gui, text=text, style='My.TButton',
                         command=lambda index=filter_index: toggle_filter(index, my_filters))
        btn.pack(side="bottom", fill="both", padx="5", pady="5")

    updated_panel = Label(gui)
    updated_panel.pack(padx=40, pady=40)

    return my_filters, gui, updated_panel


def display_updated_image(image_data, display_panel):
    """
    Display an updated image on the graphical user interface.

    Args:
        image_data (numpy.ndarray): Image data to be shown.
        display_panel (Label): Panel to showcase the image.

    Returns:
        None
    """

    updated_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)))
    display_panel.configure(image=updated_image)
    display_panel.image = updated_image


