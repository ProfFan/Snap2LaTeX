# Snap2LaTeX

Snap2LaTeX is a tool that converts a picture of a mathematical equation into a LaTeX code. It uses https://github.com/NormXU/nougat-latex-ocr to recognize the equation and convert it into LaTeX code.

Compared to LaTeX-OCR, Snap2LaTeX is a standalone application that does not require any install. It has a simpler interface and uses MPS by default.

# Usage

Download from [releases](https://github.com/ProfFan/Snap2LaTeX/releases).

Run the application.

The application will start downloading the model. It will take a few minutes.

![](./images/downloading.png)

There will be an icon in the system tray.

![](./images/dock-icon.png)

Click on it and select "Capture" to capture a screenshot of the equation. The application will then recognize the equation and display the LaTeX code. When the processing is in progress, the icon will turn yellow.

| Capture | LaTeX |
| --- | --- |
| ![](./images/latex-img.png) | ![](./images/screenshot.png) |

**The LaTeX code will be automatically copied to the clipboard**

| Matrix | Code |
| --- | --- |
| ![](./matrix.png) | ![](./images/array.png) |

# Build

Dependencies are managed with [uv](https://docs.astral.sh/uv/); exact versions
are pinned in `standalone_app/uv.lock`. Install uv, then:

```bash
cd standalone_app
uv sync                                      # create venv + install from the lockfile
uv run pyinstaller Snap2LaTeX.spec --noconfirm
```

The app bundle is written to `standalone_app/dist/Snap2LaTeX.app`. See
[`standalone_app/README.md`](./standalone_app/README.md) for details (running
from source, managing dependencies, troubleshooting).

# Alternatives

- [Mathpix](https://mathpix.com/): Paid service.
- [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR): Free and Open Source. Slightly more verbose interface.

# LICENSE

```
Copyright 2024 Fan Jiang
Copyright 2024 NormXU

Apache License 2.0
```
