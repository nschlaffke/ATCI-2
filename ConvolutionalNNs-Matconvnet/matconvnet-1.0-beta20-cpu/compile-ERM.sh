#!/bin/tcsh

### (Previous) Modify Makefile
## For CPU compilation:
#
# ENABLE_GPU ?=
#
## For GPU compilation:
#
# ENABLE_GPU ?= yes
# ENABLE_CUDNN ?= OPTIONAL
# CUDNNROOT ?= OPTIONAL
#
## Common:
#
# ARCH ?= glnxa64
# MATLABROOT ?= /home/eromero/tools/matlab-8.5.0-R2015a/
# CUDAROOT ?= /home/eromero/tools/cuda-7.5/
# 
## *** THERE ARE SOME ERRORS IN THE Makefile: Search *ERM in Makefile
#

make clean
#
make
#
