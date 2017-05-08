# Makefile
# GNU makefile for example swift-hohenberg model code
# Questions/comments to gruberja@gmail.com (Jason Gruber)

# includes
incdir = ../../../include

# compilers/flags
compiler = g++
pcompiler = mpic++
flags = -g -O2 -Wall -I $(incdir)
pflags = $(flags) -include mpi.h
links = -lz -fopenmp

# the program
swift-hohenberg: swift-hohenberg.cpp
	$(compiler) $(flags) $< -o $@ $(links)

parallel: swift-hohenberg.cpp
	$(pcompiler) $(pflags) $< -o $@ $(links)

clean:
	rm -f swift-hohenberg parallel
