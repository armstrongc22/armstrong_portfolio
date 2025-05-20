# portfolio/pages/__init__.py
import os, sys

# Get the absolute path to this pages/ folder
_BASE = os.path.dirname(__file__)

# Loop every immediate subfolder of pages/
for folder in os.listdir(_BASE):
    full = os.path.join(_BASE, folder)
    # If it's a directory, add it to Python's search path
    if os.path.isdir(full) and not folder.startswith("__"):
        sys.path.insert(0, full)