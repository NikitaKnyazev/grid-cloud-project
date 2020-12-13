#!/bin/bash
app="dfdnet"
docker build -t ${app} .
docker run -d -p 4000:4000 ${app}
