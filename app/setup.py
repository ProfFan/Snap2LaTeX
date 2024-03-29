"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

from setuptools import setup

APP = ["screencapture.py"]
DATA_FILES = ["icon.png"]
OPTIONS = {
    "plist": {"LSUIElement": True},
    "iconfile": "icon.icns",
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
    install_requires=["PyQt6", "requests", "chardet"],
)
