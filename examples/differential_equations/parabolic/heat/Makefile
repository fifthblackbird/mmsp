# Makefile
# GNU makefile for example Manufactured Solution code
# Questions/comments to trevor.keller@gmail.com (Trevor Keller)

# includes
incdir = ../../../../include

# compilers/flags
compiler = g++
pcompiler = mpic++
flags = -O3 -I $(incdir)
pflags = $(flags) -include mpi.h

# the program
heat: heat.cpp
	$(compiler) $(flags) $< -o $@ -lz

parallel: heat.cpp
	$(pcompiler) $(pflags) $< -o $@ -lz

clean:
	rm -f heat parallel
