const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))


static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
    int K_div_8 = (K>>3) << 3;

    /* For each row i of A */
    for (int i = 0; i < M; ++i) {
    /* For each column j of B */ 
        int jlda = 0;
        for (int j = 0; j < N; ++j) {
        /* Compute C(i,j) */
            double cij = C[i+j*lda];

            for (int k = 0; k < K_div_8; k+=8) {
                double cij1 = A[i+k*lda] * B[k+jlda];
                double cij2 = A[i+(k+1)*lda] * B[k+jlda+1];
                double cij3 = A[i+(k+2)*lda] * B[k+jlda+2];
                double cij4 = A[i+(k+3)*lda] * B[k+jlda+3];

                double cij5 = A[i+(k+4)*lda] * B[k+jlda+4];
                double cij6 = A[i+(k+5)*lda] * B[k+jlda+5];
                double cij7 = A[i+(k+6)*lda] * B[k+jlda+6];
                double cij8 = A[i+(k+7)*lda] * B[k+jlda+7];

                cij += cij1 + cij2 + cij3 + cij4 + cij5 + cij6 + cij7 + cij8;
                C[i+j*lda] = cij;
            }

            for (int k = K_div_8; k < K; ++k) {
                cij += A[i+k*lda] * B[k+jlda];
                C[i+jlda] = cij;
            }
            jlda += lda;
        }
    }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}
