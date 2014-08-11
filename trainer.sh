#!/bin/sh
gcc -o trainer trainer.c -lm
./trainer $1 $2 $3 $4 $5