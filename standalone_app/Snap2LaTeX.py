import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from transformers import NougatImageProcessor

from logging import info
import os
import tempfile
import webbrowser
import subprocess

import rumps
import multiprocessing as mp

VERSION = "0.6.0"

USE_FLOAT16 = False

class Snap2LaTeXApp(rumps.App):
    def __init__(self):
        # Initialize the AI model
        self.model_name = "Norm/nougat-latex-base"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.latex_processor = None
        self.USE_FLOAT16 = False
        
        # Initialize rumps app with icon and template mode
        super().__init__("Snap2LaTeX", icon="resources/snap2latex.png", quit_button=None, template=True)

        # Set up the complete menu including options submenu
        self.setup_menu()
        
        # Optionally auto-load model (comment out if you want manual loading only)
        self.load_model()
        
    def setup_menu(self):
        """Set up the complete menu structure"""
        # Create the Options submenu items
        float16_item = rumps.MenuItem("Use Float16", callback=self.toggle_float16)
        float16_item.state = 1 if self.USE_FLOAT16 else 0
        
        # Create load model item (dynamic text and state)
        if self.model is None:
            load_model_item = rumps.MenuItem("Load Model", callback=self.manual_load_model)
        else:
            load_model_item = rumps.MenuItem("Model Loaded")
            load_model_item.set_callback(None)  # Disable callback
        
        # Create the Options submenu
        options_menu = rumps.MenuItem("Options")
        options_menu.add(load_model_item)
        options_menu.add(None)  # Separator in submenu
        options_menu.add(float16_item)
        
        # Set up the complete menu structure
        self.menu = [
            "Capture",
            options_menu,
            None,  # Separator
            "Check for Updates", 
            "About",
            None,  # Separator
            "Quit"
        ]
        
    def manual_load_model(self, sender):
        """Manually load the AI model when clicked from menu"""
        self.load_model()
        # Refresh the menu to update the load model item
        self.setup_menu()
    
    def load_model(self):
        """Load the AI model"""
        try:
            rumps.notification("Snap2LaTeX", "", "Loading model...")
            
            # Load model
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name, device_map=self.device)
            
            if self.USE_FLOAT16:
                self.model = self.model.half()
                
            # Load tokenizer and processor
            self.tokenizer = NougatTokenizerFast.from_pretrained(self.model_name)
            self.latex_processor = NougatImageProcessor.from_pretrained(self.model_name)
            
            rumps.notification("Snap2LaTeX", "", "Model loaded successfully!")
            info("Model loaded.")
            
            # Update menu to reflect model loaded state
            self.setup_menu()
            
        except Exception as e:
            rumps.alert("Error", f"Failed to load model: {e}")
            self.model = None  # Reset model state on error
    
    def analyze_image(self, image_path):
        """Analyze image and convert to LaTeX"""
        try:
            if not self.model:
                rumps.alert("Error", "Model not loaded yet. Please wait.")
                return
                
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            pixel_values = self.latex_processor(image, return_tensors="pt").pixel_values

            if self.USE_FLOAT16:
                pixel_values = pixel_values.half()

            decoder_input_ids = self.tokenizer(
                self.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
            ).input_ids
            
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values.to(self.device),
                    decoder_input_ids=decoder_input_ids.to(self.device),
                    max_length=self.model.decoder.config.max_length,
                    early_stopping=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[self.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )
            
            sequence = self.tokenizer.batch_decode(outputs.sequences)[0]
            sequence = (
                sequence.replace(self.tokenizer.eos_token, "")
                .replace(self.tokenizer.pad_token, "")
                .replace(self.tokenizer.bos_token, "")
            )

            info(f"${sequence}$")

            # Copy to clipboard
            subprocess.run(['pbcopy'], input=sequence.encode('utf-8'))
            
            # Show result
            rumps.alert("LaTeX Result", f"Copied to clipboard:\n\n{sequence}")

        except Exception as e:
            rumps.alert("Error", f"An error occurred: {e}")
            raise e

    @rumps.clicked("Capture")
    def capture(self, _):
        """Capture screenshot and analyze"""
        try:
            # Create temp file
            temp_dir = tempfile.mkdtemp("snap2latex")
            temp_file = os.path.join(temp_dir, "capture.png")
            
            # Capture screen interactively
            result = subprocess.run(f"screencapture -i -Jselection {temp_file}", shell=True)
            
            if os.path.exists(temp_file):
                rumps.notification("Snap2LaTeX", "", "Processing image...")
                self.analyze_image(temp_file)
                os.remove(temp_file)
            
            os.rmdir(temp_dir)
            
        except Exception as e:
            rumps.alert("Error", f"Capture failed: {e}")
    
    def toggle_float16(self, sender):
        """Toggle Float16 mode"""
        self.USE_FLOAT16 = not self.USE_FLOAT16
        sender.state = 1 if self.USE_FLOAT16 else 0
        
        if self.USE_FLOAT16:
            rumps.notification("Snap2LaTeX", "", "Switching to Float16 mode...")
            if self.model:
                self.model.half()
        else:
            rumps.notification("Snap2LaTeX", "", "Switching to Float32 mode...")
            # Reload model
            self.load_model()
        
        # Refresh menu to update states
        self.setup_menu()

    @rumps.clicked("Check for Updates")
    def check_for_updates(self, _):
        """Check for updates"""
        try:
            import requests
            import json

            response = requests.get(
                "https://api.github.com/repos/ProfFan/Snap2LaTeX/releases/latest"
            )
            response.raise_for_status()
            data = response.json()
            latest_version = data["tag_name"]
            
            if latest_version != VERSION:
                response = rumps.alert(
                    "Update Available",
                    f"A new version of Snap2LaTeX is available: {latest_version}\n\nWould you like to open the releases page?",
                    ok="Open",
                    cancel="Cancel"
                )
                if response == 1:  # OK button clicked
                    webbrowser.open("https://github.com/ProfFan/Snap2LaTeX/releases")
            else:
                rumps.alert("Up to Date", "Snap2LaTeX is up to date.")
                
        except Exception as e:
            rumps.alert("Error", f"Failed to check for updates: {e}")

    @rumps.clicked("About")
    def about(self, _):
        """Show about dialog"""
        about_text = f"""Snap2LaTeX ©2024 Fan Jiang Version {VERSION}

Snap2LaTeX is a tool that converts a picture of a mathematical equation into a LaTeX code.

Check for Updates at: https://github.com/ProfFan/Snap2LaTeX/releases

Model by @NormXU: https://github.com/NormXU/nougat-latex-ocr"""
        
        rumps.alert("About Snap2LaTeX", about_text)

    @rumps.clicked("Quit")
    def quit_app(self, _):
        """Quit the application"""
        rumps.quit_application()


if __name__ == "__main__":
    mp.freeze_support()
    Snap2LaTeXApp().run()
