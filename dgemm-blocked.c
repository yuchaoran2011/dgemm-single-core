const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C) {
    int M_div_8 = (M>>3) << 3;

    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {  

            double bkj = B[k+j*lda];
            int klda = k*lda;
            int jlda = j*lda;

            for (int i = 0; i < M_div_8; i+=8) {
                C[i+jlda] += A[i+klda] * bkj; 
                C[1+i+jlda] += A[i+klda+1] * bkj;
                C[2+i+jlda] += A[i+klda+2] * bkj;
                C[3+i+jlda] += A[i+klda+3] * bkj;

                C[4+i+jlda] += A[i+klda+4] * bkj;
                C[5+i+jlda] += A[i+klda+5] * bkj;
                C[6+i+jlda] += A[i+klda+6] * bkj;
                C[7+i+jlda] += A[i+klda+7] * bkj;
            }

            for (int i=M_div_8; i<M; ++i) {
                C[i+j*lda] += A[i+klda] * bkj;
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C) {
    /* For each block-row of A */ 
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
    /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
        /* Accumulate block dgemms into block of C */
            for (int k = 0; k < lda; k += BLOCK_SIZE)   {
	       /* Correct block dimensions if block "goes off edge of" the matrix */
	           int M = min (BLOCK_SIZE, lda-i);
	           int N = min (BLOCK_SIZE, lda-j);
	           int K = min (BLOCK_SIZE, lda-k);

	           /* Perform individual block dgemm */
	           do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
            }
        }
    }
}