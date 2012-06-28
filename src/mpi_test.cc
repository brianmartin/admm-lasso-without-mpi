#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);

	int i;
	int n = 5;
	double *send;
	double *recv;

	send = (double*) malloc(n * sizeof(double));
	recv = (double*) malloc(n * sizeof(double));

	for (i = 0; i < n; i++) {
		send[i] = (double) i;
	}

	MPI_Allreduce((void*) send, (void*) recv, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		for (i = 0; i < n; i++) {
			cout << (double) recv[i];
			cout << "\n";
		}
	}

	MPI_Finalize();

	free((void*) send);
	free((void*) recv);
}
