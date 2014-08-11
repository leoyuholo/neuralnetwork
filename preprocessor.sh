#!/bin/sh
gcc -o preprocess preprocess.c -Wall -lm
./preprocess $1 $2 $3