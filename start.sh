#!/bin/sh
python3 src/misc_funcs.py
jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root & python3 app.py
