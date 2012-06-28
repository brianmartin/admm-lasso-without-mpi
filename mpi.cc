#include <string>
#include "allreduce.h"
#include <iostream>

using namespace std;

size_t unique_id = 0;
size_t rank;
size_t total;

void MPI_Init(int* argc, char*** argv) {
	rank = atoi((*argv)[1]);
	total = atoi((*argv)[2]);
    return;
}

void MPI_Comm_rank(int mpi_comm, int* rank_out) {
    *rank_out = rank;
}

void MPI_Comm_size(int mpi_comm, int* size_out) {
    *size_out = total;
}

void MPI_Allreduce(double* send, double* recv, int size, int mpi_type, int mpi_op, int mpi_comm) {
	std::string master_location = "localhost";

	//copy to float array
	float* buffer = (float*) malloc(size * sizeof(float));
	int i;
	for (i = 0; i < size; i++) {
		buffer[i] = (float) send[i];
	}

	// call the VW all_reduce
    all_reduce(buffer, size, master_location, unique_id, total, rank);

	//copy back to recv
	for (i = 0; i < size; i++) {
		recv[i] = (double) buffer[i];
	}

    return;
}

void MPI_Finalize() {
	// do nothing
    return;
}