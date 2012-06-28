#ifndef MPI_H
#define MPI_H

int MPI_COMM_WORLD = 0;
int MPI_DOUBLE = 0;
int MPI_SUM = 0;

void MPI_Init(int* argc, char*** argv);
int MPI_Comm_rank(int mpi_comm, int* rank_out);
int MPI_Comm_size(int mpi_comm, int* size_out);
void MPI_Allreduce();
void MPI_Allreduce(double* send, double* recv, int num, int mpi_type, int mpi_op, int mpi_comm);
void MPI_Finalize();

#endif
