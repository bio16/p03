.PHONY: all informe figs
src=informe

default: all
	
informe:
	pdflatex ${src}.tex

figs:
	# generar figs para parte d) (no demora nada)
	cd ../parte_d && ./gg.py && ./sort_sex.py && cd -
	cd .. && python3 mutual_info.py && cd -


all: figs informe

#EOF
