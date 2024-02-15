import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from transformers import NougatImageProcessor
import accelerate

import logging
from logging import info
import io

from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

if __name__ == "__main__":

    model_name = "Norm/nougat-latex-base"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # init model
    model = VisionEncoderDecoderModel.from_pretrained(model_name, device_map=device)

    # init processor
    tokenizer = NougatTokenizerFast.from_pretrained(model_name)

    latex_processor = NougatImageProcessor.from_pretrained(model_name)

    info("Loaded model.")

    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)

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
