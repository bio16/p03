#!/bin/bash
src=informe

default:
	pdflatex ${src}.tex

all:
	# generar figs para parte d) (no demora nada)
	cd ../parte_d && ./gg.py && ./sort_sex.py && cd -
	pdflatex ${src}.tex

#EOF
