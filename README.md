# Snap2LaTeX

Snap2LaTeX is a tool that converts a picture of a mathematical equation into a LaTeX code. It uses https://github.com/NormXU/nougat-latex-ocr to recognize the equation and convert it into LaTeX code.

# Usage

Download from [releases](https://github.com/ProfFan/Snap2LaTeX/releases).

Run the application.

There will be an icon in the system tray.

![](./images/dock-icon.png)

Click on it and select "Capture" to capture a screenshot of the equation. The application will then recognize the equation and display the LaTeX code.

![](./images/latex-img.png)

![](./images/screenshot.png)

Double click on the LaTeX code and right click to copy.

| Matrix | Code |
| --- | --- |
| ![](./matrix.png) | ![](./images/array.png) |

# Build

```bash
pip install -U https://github.com/huggingface/transformers/archive/refs/heads/main.zip
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu -U
pip install -r requirements.txt
cd standalone_app
pyinstaller Snap2LaTeX.spec
```

# LICENSE

```
Copyright 2024 Fan Jiang
Copyright 2024 NormXU

Apache License 2.0
```
