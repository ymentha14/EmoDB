#!/bin/sh
docker run -it -p 127.0.0.1:5000:5000 -p 8887:8888 user/emodb /bin/bash
