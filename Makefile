CC=g++
CFLAGS=-lOpenCL -Wall -O2 -lm

default: all

# modify to generate parallel and optimized versions!
all: bin bin/nbody-seq

part1: bin bin/nbody-opencl

part2: bin bin/nbody-part2

report: report.pdf

bin:
	mkdir bin

bin/nbody-seq: src/nbody-seq.c
	$(CC) $(CFLAGS) -o bin/nbody-seq $<

bin/nbody-opencl: src/nbody-opencl.c
	$(CC) $(CFLAGS) -o bin/nbody-opencl $<

bin/nbody-part2: src/nbody-part2.c
	$(CC) $(CFLAGS) -o bin/nbody-part2 $<

report.pdf: report/report.tex
	cd report && pdflatex report.tex && pdflatex report.tex
	mv report/report.pdf report.pdf

clean:
	rm -r bin
