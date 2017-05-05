# Makefile
# GNU makefile for example swift-hohenberg model code
# Questions/comments to gruberja@gmail.com (Jason Gruber)

# includes
incdir = ../../../include

# compilers/flags
compiler = g++
pcompiler = mpic++
flags = -g -O2 -I $(incdir)
pflags = $(flags) -include mpi.h

# the program
swift-hohenberg: swift-hohenberg.cpp
	$(compiler) $(flags) $< -o $@ -lz

parallel: swift-hohenberg.cpp
	$(pcompiler) $(pflags) $< -o $@ -lz

clean:
	rm -f swift-hohenberg parallel
