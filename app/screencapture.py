import chardet

from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

app = QApplication([])
app.setQuitOnLastWindowClosed(False)

# Create the icon
icon = QIcon("icon.png")

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
    # send the image to the server with a form POST request
    import requests
    try:
        files = {"image": open(temp_file, "rb")}
        response = requests.post("http://localhost:8000", files=files)
        response.raise_for_status()
        # get the response
        data = response.json()
        # show the response in a dialog
        dialog = QMessageBox()
        dialog.setText(data["latex"])
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

# Add a Quit option to the menu.
quit = QAction("Quit")
quit.triggered.connect(app.quit)
menu.addAction(quit)

# Add the menu to the tray
tray.setContextMenu(menu)

app.exec()