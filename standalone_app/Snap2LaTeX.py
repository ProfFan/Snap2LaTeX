import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from transformers import NougatImageProcessor
import accelerate

import re
import logging
from logging import info
import io
import sys
from multiprocessing.queues import Queue

from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QTimer, QUrl

import multiprocessing as mp

VERSION = "0.4.0"


class StdoutQueue(Queue):
    def __init__(self, maxsize=-1, block=True, timeout=None):
        self.block = block
        self.timeout = timeout
        super().__init__(maxsize, ctx=mp.get_context())

    def write(self, msg):
        self.put(msg)

    def flush(self):
        sys.__stdout__.flush()


def load_model_proc(model_name, q: StdoutQueue):
    sys.stdout = q
    sys.stderr = q
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # init model
    model = VisionEncoderDecoderModel.from_pretrained(model_name, device_map=device)

    q.close()


def app_show_progress(model_name):
    app = QApplication([])

    q = StdoutQueue()
    load_process = mp.Process(target=load_model_proc, args=(model_name, q))
    load_process.start()

    # Progress window
    progress = QProgressDialog()
    progress.setLabelText("Loading model...")
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    # disable the cancel button
    progress.setCancelButton(None)
    progress.setWindowFlags(
        Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint
    )

    while load_process.is_alive():
        while not q.empty():
            # match the ":   5%|" pattern with the regex
            # and extract the percentage
            match = re.search(r":\s+(\d+)%\|", q.get())
            if match:
                progress.setValue(int(match.group(1)))

            # append the message to the progress window
            progress.setLabelText(q.get())
        app.processEvents()

    print("Model loaded.")

    load_process.join()
    progress.close()
    app.quit()
    print("Model check complete.")

USE_FLOAT16 = False

if __name__ == "__main__":
    mp.freeze_support()

    model_name = "Norm/nougat-latex-base"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    app_show_progress(model_name)

    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    clipboard = app.clipboard()

    # init model
    model = VisionEncoderDecoderModel.from_pretrained(model_name, device_map=device)

    # convert to float16
    if USE_FLOAT16:
        model = model.half()

    # init processor
    tokenizer = NougatTokenizerFast.from_pretrained(model_name)

    latex_processor = NougatImageProcessor.from_pretrained(model_name)

    info("Loaded model.")

    # Create the icon
    from os import path

    path_to_icon = path.abspath(path.join(path.dirname(__file__), "icon.png"))
    icon = QIcon(path_to_icon)
    path_to_inproc_icon = path.abspath(path.join(path.dirname(__file__), "inproc.png"))
    icon_inproc = QIcon(path_to_inproc_icon)
    icon_pixmap = QPixmap(path_to_icon)

    # Create the tray
    tray = QSystemTrayIcon()
    tray.setIcon(icon)
    tray.setVisible(True)
    tray.setObjectName("Image2LaTeX")

    def analyze_image(temp_file, temp_dir):
        import os

        try:
            image = Image.open(temp_file)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            pixel_values = latex_processor(image, return_tensors="pt").pixel_values

            if USE_FLOAT16:
                pixel_values = pixel_values.half()

            decoder_input_ids = tokenizer(
                tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
            ).input_ids
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values.to(device),
                    decoder_input_ids=decoder_input_ids.to(device),
                    max_length=model.decoder.config.max_length,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=5,
                    bad_words_ids=[[tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )
            sequence = tokenizer.batch_decode(outputs.sequences)[0]
            sequence = (
                sequence.replace(tokenizer.eos_token, "")
                .replace(tokenizer.pad_token, "")
                .replace(tokenizer.bos_token, "")
            )

            info(f"${sequence}$")

            # show the response in a dialog
            dialog = QMessageBox()
            dialog.setIconPixmap(icon_pixmap)
            dialog.setText(sequence)
            clipboard.setText(sequence)
            dialog.exec()

        except Exception as e:
            dialog = QMessageBox()
            dialog.setIconPixmap(icon_pixmap)
            dialog.setText(f"An error occurred: {e}")
            dialog.exec()
        finally:
            os.remove(temp_file)
            os.rmdir(temp_dir)
            tray.setIcon(icon)

    def capture():
        # capture the screen interactively
        import os

        # make a temp folder
        import tempfile

        temp_dir = tempfile.mkdtemp("image2latex")
        temp_file = os.path.join(temp_dir, "capture.png")
        os.system(f"screencapture -i -Jselection {temp_file}")

        # set the qicon to a processing icon
        tray.setIcon(icon_inproc)

        QTimer.singleShot(1, lambda: analyze_image(temp_file, temp_dir))

    # Create the menu
    menu = QMenu()
    action = QAction("Capture")
    action.triggered.connect(capture)
    menu.addAction(action)

    # Add an "Options" submenu
    options = QMenu("Options")
    menu.addMenu(options)

    # Add a "Use Float16" checkbox to the "Options" submenu
    use_float16 = QAction("Use Float16", checkable=True)
    use_float16.setChecked(USE_FLOAT16)

    def set_use_float16(checked):
        global model
        global USE_FLOAT16
        USE_FLOAT16 = checked
        if checked:
            print("Using float16.")
            model.half()
        else:
            print("Using float32.")
            # reload the model
            model = VisionEncoderDecoderModel.from_pretrained(model_name, device_map=device)
    use_float16.triggered.connect(set_use_float16)
    options.addAction(use_float16)

    # Add a "Check for Updates" option to the menu
    def check_for_updates():
        import requests
        import json

        try:
            response = requests.get(
                "https://api.github.com/repos/ProfFan/Snap2LaTeX/releases/latest"
            )
            response.raise_for_status()
            data = response.json()
            latest_version = data["tag_name"]
            if latest_version != VERSION:
                dialog = QMessageBox()
                dialog.setIconPixmap(icon_pixmap)
                dialog.setText(
                    f"A new version of Snap2LaTeX is available: {latest_version}"
                )
                # add a button to open the release page and a button to close the dialog
                open_button = dialog.addButton(
                    "Open", QMessageBox.ButtonRole.ActionRole
                )
                dialog.addButton(QMessageBox.StandardButton.Close)

                # open the release page when the open button is clicked
                open_button.clicked.connect(
                    lambda: QDesktopServices.openUrl(
                        QUrl("https://github.com/ProfFan/Snap2LaTeX/releases")
                    )
                )

                dialog.exec()
            else:
                dialog = QMessageBox()
                dialog.setIconPixmap(icon_pixmap)
                dialog.setText("Snap2LaTeX is up to date.")
                dialog.exec()
        except requests.RequestException as e:
            dialog = QMessageBox()
            dialog.setIconPixmap(icon_pixmap)
            dialog.setText(f"An error occurred: {e}")
            dialog.exec()

    check_updates = QAction("Check for Updates")
    check_updates.triggered.connect(check_for_updates)
    menu.addAction(check_updates)

    # Add About option to the menu
    def about_window():
        dialog = QMessageBox()
        dialog.setText(
            f"""Snap2LaTeX ©2024 Fan Jiang Version {VERSION}\n
Snap2LaTeX is a tool that converts a picture of a mathematical equation into a LaTeX code.\n
Check for Updates at: https://github.com/ProfFan/Snap2LaTeX/releases \n
Model by @NormXU: https://github.com/NormXU/nougat-latex-ocr.
        """
        )
        # set dialog pixmap
        dialog.setIconPixmap(icon_pixmap)
        dialog.exec()

    about = QAction("About")
    about.triggered.connect(about_window)
    menu.addAction(about)

    # Add a Quit option to the menu.
    quit = QAction("Quit")
    quit.triggered.connect(app.quit)
    menu.addAction(quit)

    # Add the menu to the tray
    tray.setContextMenu(menu)

    app.exec()
