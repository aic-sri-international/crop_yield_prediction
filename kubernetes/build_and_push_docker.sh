#!/bin/bash

docker build . -t worldmodelers.cse.sri.com/crop-yield:v1.2
docker push worldmodelers.cse.sri.com/crop-yield:v1.2
