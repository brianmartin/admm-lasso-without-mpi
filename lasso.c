/*
 * Solve a distributed lasso problem, i.e.,
 *
 *   minimize  (1/2)||Ax - b||_2^2 + lambda*||x||_1.
 *
 * The implementation uses MPI for distributed communication
 * and the GNU Scientific Library (GSL) for math.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mmio.h"
#include "mpi.h"
#include "allreduce.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

void soft_threshold(gsl_vector *v, double k);
double objective(gsl_matrix *A, gsl_vector *b, double lambda, gsl_vector *z);

int main(int argc, char **argv) {
	const int MAX_ITER  = 50;
	const double RELTOL = 1e-2;
	const double ABSTOL = 1e-4;

    int i;

	/*
	 * Some bookkeeping variables for MPI. The 'rank' of a process is its numeric id
	 * in the process pool. For example, if we run a program via `mpirun -np 4 foo', then
	 * the process ranks are 0 through 3. Here, N and size are the total number of processes
	 * running (in this example, 4).
	 */

    MPI_Init(&argc, &argv);

	int rank = 0;
	int size = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Determine current running process
	MPI_Comm_size(MPI_COMM_WORLD, &size); // Total number of processes
	double N = (double) size;             // Number of subsystems/slaves for ADMM

	/* Read in local data */

	int skinny;           // A flag indicating whether the matrix A is fat or skinny
	FILE *f;
	int m, n;
	int row, col;
	double entry;

	/*
	 * Subsystem n will look for files called An.dat and bn.dat
	 * in the current directory; these are its local data and do not need to be
	 * visible to any other processes. Note that
	 * m and n here refer to the dimensions of the *local* coefficient matrix.
	 */

	/* Read A */
	char s[20];
	sprintf(s, "data/A%d.dat", rank + 1);
	printf("[%d] reading %s\n", rank, s);

	f = fopen(s, "r");
	if (f == NULL) {
		printf("[%d] ERROR: %s does not exist, exiting.\n", rank, s);
		exit(EXIT_FAILURE);
	}
	mm_read_mtx_array_size(f, &m, &n);
	gsl_matrix *A = gsl_matrix_calloc(m, n);
	for (i = 0; i < m*n; i++) {
		row = i % m;
		col = floor(i/m);
		fscanf(f, "%lf", &entry);
		gsl_matrix_set(A, row, col, entry);
	}
	fclose(f);

	/* Read b */
	sprintf(s, "data/b%d.dat", rank + 1);
	printf("[%d] reading %s\n", rank, s);

	f = fopen(s, "r");
	if (f == NULL) {
		printf("[%d] ERROR: %s does not exist, exiting.\n", rank, s);
		exit(EXIT_FAILURE);
	}
	mm_read_mtx_array_size(f, &m, &n);
	gsl_vector *b = gsl_vector_calloc(m);
	for (i = 0; i < m; i++) {
		fscanf(f, "%lf", &entry);
		gsl_vector_set(b, i, entry);
	}
	fclose(f);

	m = A->size1;
	n = A->size2;
	skinny = (m >= n);

	/*
	 * These are all variables related to ADMM itself. There are many
	 * more variables than in the Matlab implementation because we also
	 * require vectors and matrices to store various intermediate results.
	 * The naming scheme follows the Matlab version of this solver.
	 */

	double rho = 1.0;

	gsl_vector *x      = gsl_vector_calloc(n);
	gsl_vector *u      = gsl_vector_calloc(n);
	gsl_vector *z      = gsl_vector_calloc(n);
	gsl_vector *y      = gsl_vector_calloc(n);
	gsl_vector *r      = gsl_vector_calloc(n);
	gsl_vector *zprev  = gsl_vector_calloc(n);
	gsl_vector *zdiff  = gsl_vector_calloc(n);

	gsl_vector *q      = gsl_vector_calloc(n);
	gsl_vector *w      = gsl_vector_calloc(n);
	gsl_vector *Aq     = gsl_vector_calloc(m);
	gsl_vector *p      = gsl_vector_calloc(m);

	gsl_vector *Atb    = gsl_vector_calloc(n);

	double send[3]; // an array used to aggregate 3 scalars at once
	double recv[3]; // used to receive the results of these aggregations

	double nxstack  = 0;
	double nystack  = 0;
	double prires   = 0;
	double dualres  = 0;
	double eps_pri  = 0;
	double eps_dual = 0;

	/* Precompute and cache factorizations */

	gsl_blas_dgemv(CblasTrans, 1, A, b, 0, Atb); // Atb = A^T b

	/*
	 * The lasso regularization parameter here is just hardcoded
	 * to 0.5 for simplicity. Using the lambda_max heuristic would require
	 * network communication, since it requires looking at the *global* A^T b.
	 */

	double lambda = 0.5;
	if (rank == 0) {
		printf("using lambda: %.4f\n", lambda);
	}

	gsl_matrix *L;

	/* Use the matrix inversion lemma for efficiency; see section 4.2 of the paper */
	if (skinny) {
		/* L = chol(AtA + rho*I) */
		L = gsl_matrix_calloc(n,n);

		gsl_matrix *AtA = gsl_matrix_calloc(n,n);
		gsl_blas_dsyrk(CblasLower, CblasTrans, 1, A, 0, AtA);

		gsl_matrix *rhoI = gsl_matrix_calloc(n,n);
		gsl_matrix_set_identity(rhoI);
		gsl_matrix_scale(rhoI, rho);

		gsl_matrix_memcpy(L, AtA);
		gsl_matrix_add(L, rhoI);
		gsl_linalg_cholesky_decomp(L);

		gsl_matrix_free(AtA);
		gsl_matrix_free(rhoI);
	} else {
		/* L = chol(I + 1/rho*AAt) */
		L = gsl_matrix_calloc(m,m);

		gsl_matrix *AAt = gsl_matrix_calloc(m,m);
		gsl_blas_dsyrk(CblasLower, CblasNoTrans, 1, A, 0, AAt);
		gsl_matrix_scale(AAt, 1/rho);

		gsl_matrix *eye = gsl_matrix_calloc(m,m);
		gsl_matrix_set_identity(eye);

		gsl_matrix_memcpy(L, AAt);
		gsl_matrix_add(L, eye);
		gsl_linalg_cholesky_decomp(L);

		gsl_matrix_free(AAt);
		gsl_matrix_free(eye);
	}

	/* Main ADMM solver loop */

	int iter = 0;
	if (rank == 0) {
		printf("%3s %10s %10s %10s %10s %10s\n", "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective");
	}

	while (iter < MAX_ITER) {

		/* u-update: u = u + x - z */
		gsl_vector_sub(x, z);
		gsl_vector_add(u, x);

		/* x-update: x = (A^T A + rho I) \ (A^T b + rho z - y) */
		gsl_vector_memcpy(q, z);
		gsl_vector_sub(q, u);
		gsl_vector_scale(q, rho);
		gsl_vector_add(q, Atb);   // q = A^T b + rho*(z - u)

		if (skinny) {
			/* x = U \ (L \ q) */
			gsl_linalg_cholesky_solve(L, q, x);
		} else {
			/* x = q/rho - 1/rho^2 * A^T * (U \ (L \ (A*q))) */
			gsl_blas_dgemv(CblasNoTrans, 1, A, q, 0, Aq);
			gsl_linalg_cholesky_solve(L, Aq, p);
			gsl_blas_dgemv(CblasTrans, 1, A, p, 0, x); /* now x = A^T * (U \ (L \ (A*q)) */
			gsl_vector_scale(x, -1/(rho*rho));
			gsl_vector_scale(q, 1/rho);
			gsl_vector_add(x, q);
		}

		/*
		 * Message-passing: compute the global sum over all processors of the
		 * contents of w and t. Also, update z.
		 */

		gsl_vector_memcpy(w, x);
		gsl_vector_add(w, u);   // w = x + u

		gsl_blas_ddot(r, r, &send[0]);
		gsl_blas_ddot(x, x, &send[1]);
		gsl_blas_ddot(u, u, &send[2]);
		send[2] /= pow(rho, 2);

		gsl_vector_memcpy(zprev, z);

		// could be reduced to a single Allreduce call by concatenating send to w
		MPI_Allreduce(w->data, z->data,  n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(send,    recv,     3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		prires  = sqrt(recv[0]);  /* sqrt(sum ||r_i||_2^2) */
		nxstack = sqrt(recv[1]);  /* sqrt(sum ||x_i||_2^2) */
		nystack = sqrt(recv[2]);  /* sqrt(sum ||y_i||_2^2) */

		gsl_vector_scale(z, 1/N);
		soft_threshold(z, lambda/(N*rho));

		/* Termination checks */

		/* dual residual */
		gsl_vector_memcpy(zdiff, z);
		gsl_vector_sub(zdiff, zprev);
		dualres = sqrt(N) * rho * gsl_blas_dnrm2(zdiff); /* ||s^k||_2^2 = N rho^2 ||z - zprev||_2^2 */

		/* compute primal and dual feasibility tolerances */
		eps_pri  = sqrt(n*N)*ABSTOL + RELTOL * fmax(nxstack, sqrt(N)*gsl_blas_dnrm2(z));
		eps_dual = sqrt(n*N)*ABSTOL + RELTOL * nystack;

		if (rank == 0) {
			printf("%3d %10.4f %10.4f %10.4f %10.4f %10.4f\n", iter,
					prires, eps_pri, dualres, eps_dual, objective(A, b, lambda, z));
		}

		if (prires <= eps_pri && dualres <= eps_dual) {
			break;
		}

		/* Compute residual: r = x - z */
		gsl_vector_memcpy(r, x);
		gsl_vector_sub(r, z);

		iter++;
	}

	/* Have the master write out the results to disk */
	if (rank == 0) {
		f = fopen("data/solution.dat", "w");
		gsl_vector_fprintf(f, z, "%lf");
		fclose(f);
	}

	//MPI_Finalize(); /* Shut down the MPI execution environment */

	/* Clear memory */
	gsl_matrix_free(A);
	gsl_matrix_free(L);
	gsl_vector_free(b);
	gsl_vector_free(x);
	gsl_vector_free(u);
	gsl_vector_free(z);
	gsl_vector_free(y);
	gsl_vector_free(r);
	gsl_vector_free(w);
	gsl_vector_free(zprev);
	gsl_vector_free(zdiff);
	gsl_vector_free(q);
	gsl_vector_free(Aq);
	gsl_vector_free(Atb);
	gsl_vector_free(p);

	return EXIT_SUCCESS;
}

double objective(gsl_matrix *A, gsl_vector *b, double lambda, gsl_vector *z) {
	double obj = 0;
	gsl_vector *Azb = gsl_vector_calloc(A->size1);
	gsl_blas_dgemv(CblasNoTrans, 1, A, z, 0, Azb);
	gsl_vector_sub(Azb, b);
	double Azb_nrm2;
	gsl_blas_ddot(Azb, Azb, &Azb_nrm2);
	obj = 0.5 * Azb_nrm2 + lambda * gsl_blas_dasum(z);
	gsl_vector_free(Azb);
	return obj;
}

void soft_threshold(gsl_vector *v, double k) {
	double vi;
    unsigned int i;
	for (i = 0; i < v->size; i++) {
		vi = gsl_vector_get(v, i);
		if (vi > k)       { gsl_vector_set(v, i, vi - k); }
		else if (vi < -k) { gsl_vector_set(v, i, vi + k); }
		else              { gsl_vector_set(v, i, 0); }
	}
}
