#!/bin/bash
app="myapp"
docker build -t ${app} .
docker run -d -p 5000:5000 ${app}
