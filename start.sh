#!/bin/sh
jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root & python3 app.py
