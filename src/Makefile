GSLROOT=/usr/local/Cellar/gsl/1.15
BOOST_ROOT=/usr/local/Cellar/boost/1.49.0
# use this if on 64-bit machine with 64-bit GSL libraries
ARCH=x86_64
# use this if on 32-bit machine with 32-bit GSL libraries
# ARCH=i386

MPICC=mpic++
CC=g++
CFLAGS=-Wall -arch $(ARCH) -I$(GSLROOT)/include
LDFLAGS=-L$(GSLROOT)/lib -lgsl -lgslcblas -lm -L$(BOOST_ROOT)/lib -lboost_program_options-mt

all: lasso spanning_tree

lasso: lasso.o mmio.o mpi.o allreduce.o
	$(CC) $(CFLAGS) $(LDFLAGS) lasso.o mmio.o mpi.o allreduce.o -o ../lasso

test: mpi_test.o
	$(MPICC) $(CFLAGS) $(LDFLAGS) mpi_test.o -o ../mpi_test

%.o:	 %.cc  %.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o:	 %.cc
	$(CC) $(CFLAGS) -c $< -o $@

spanning_tree: spanning_tree.o
	$(CXX) $(FLAGS) -o ../$@ $+ 

clean:
	rm -vf *.o
