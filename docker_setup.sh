#!/bin/bash

echo "Installing languagemodels package...."

# Install package in development mode. Binaries will be in /home/$USER/.local/bin
pip install -e . --user

echo "Done."