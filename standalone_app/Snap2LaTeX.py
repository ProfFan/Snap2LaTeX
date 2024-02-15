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
from PyQt6.QtCore import Qt

import multiprocessing as mp


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


if __name__ == "__main__":
    mp.freeze_support()

    model_name = "Norm/nougat-latex-base"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    app_show_progress(model_name)

    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)

    # init model
    model = VisionEncoderDecoderModel.from_pretrained(model_name, device_map=device)

    # init processor
    tokenizer = NougatTokenizerFast.from_pretrained(model_name)

    latex_processor = NougatImageProcessor.from_pretrained(model_name)

    info("Loaded model.")

    # Create the icon
    from os import path

    path_to_icon = path.abspath(path.join(path.dirname(__file__), "icon.png"))
    icon = QIcon(path_to_icon)

    # Create the tray
    tray = QSystemTrayIcon()
    tray.setIcon(icon)
    tray.setVisible(True)
    tray.setObjectName("Image2LaTeX")

    def capture():
        # capture the screen interactively
        import os

        # make a temp folder
        import tempfile

        temp_dir = tempfile.mkdtemp("image2latex")
        temp_file = os.path.join(temp_dir, "capture.png")
        os.system(f"screencapture -i -Jselection {temp_file}")

        try:
            image = Image.open(temp_file)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            pixel_values = latex_processor(image, return_tensors="pt").pixel_values

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
            dialog.setText(sequence)
            dialog.exec()
        except Exception as e:
            dialog = QMessageBox()
            dialog.setText(f"An error occurred: {e}")
            dialog.exec()
        finally:
            os.remove(temp_file)
            os.rmdir(temp_dir)

    # Create the menu
    menu = QMenu()
    action = QAction("Capture")
    action.triggered.connect(capture)
    menu.addAction(action)

    # Add About option to the menu
    def about_window():
        dialog = QMessageBox()
        dialog.setText(
            """Snap2LaTeX ©2024 Fan Jiang
Snap2LaTeX is a tool that converts a picture of a mathematical equation into a LaTeX code. Model by @NormXU: https://github.com/NormXU/nougat-latex-ocr.
        """
        )
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
