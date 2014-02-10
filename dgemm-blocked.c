#include <emmintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))



/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
 /*
static void do_block (int lda, int M, int N, int K, const double* A, const double* B, double* restrict C) {
    int M_div_8 = (M>>3) << 3;

    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {  

            double bkj = B[k+j*lda];
            int klda = k*lda;
            int jlda = j*lda;

            for (int i = 0; i < M_div_8; i+=8) {
                int idx1 = i+jlda;
                int idx2 = 1+i+jlda;
                int idx3 = 2+i+jlda;
                int idx4 = 3+i+jlda;
                int idx5 = 4+i+jlda;
                int idx6 = 5+i+jlda;
                int idx7 = 6+i+jlda;
                int idx8 = 7+i+jlda;

                double cij1 = C[idx1];
                double cij2 = C[idx2];
                double cij3 = C[idx3];
                double cij4 = C[idx4];
                double cij5 = C[idx5];
                double cij6 = C[idx6];
                double cij7 = C[idx7];
                double cij8 = C[idx8];

                cij1 += A[i+klda] * bkj; 
                cij2 += A[i+klda+1] * bkj;
                cij3 += A[i+klda+2] * bkj;
                cij4 += A[i+klda+3] * bkj;
                cij5 += A[i+klda+4] * bkj; 
                cij6 += A[i+klda+5] * bkj;
                cij7 += A[i+klda+6] * bkj;
                cij8 += A[i+klda+7] * bkj;

                C[idx1] = cij1;
                C[idx2] = cij2;
                C[idx3] = cij3;
                C[idx4] = cij4;
                C[idx5] = cij5;
                C[idx6] = cij6;
                C[idx7] = cij7;
                C[idx8] = cij8;
            }

            for (int i=M_div_8; i<M; ++i) {
                C[i+j*lda] += A[i+klda] * bkj;
            }
        }
    }
}
*/


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

 /* With Intel Intrinsics, does not work yet.*/
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C) {
    int M_div_8 = (M>>3) << 3;

    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {  

            double bkj = B[k+j*lda];
            int klda = k*lda;
            int jlda = j*lda;
            __m128d bkjv = _mm_load1_pd(B+k+jlda);

            for (int i = 0; i < M_div_8; i+=8) {
                __m128d a1 = _mm_loadu_pd(A+i+klda);
                __m128d a2 = _mm_loadu_pd(A+i+klda+2);
                __m128d a3 = _mm_loadu_pd(A+i+klda+4);
                __m128d a4 = _mm_loadu_pd(A+i+klda+6);

                __m128d c1 = _mm_loadu_pd(C+i+jlda);
                __m128d c2 = _mm_loadu_pd(C+i+jlda+2);
                __m128d c3 = _mm_loadu_pd(C+i+jlda+4);
                __m128d c4 = _mm_loadu_pd(C+i+jlda+6);

                __m128d p1 = _mm_mul_pd(a1, bkjv);
                __m128d p2 = _mm_mul_pd(a2, bkjv);
                __m128d p3 = _mm_mul_pd(a3, bkjv);
                __m128d p4 = _mm_mul_pd(a4, bkjv);

                c1 = _mm_add_pd(p1, c1);
                c2 = _mm_add_pd(p2, c2);
                c3 = _mm_add_pd(p3, c3);
                c4 = _mm_add_pd(p4, c4);

                _mm_storeu_pd(C+i+jlda, c1);
                _mm_storeu_pd(C+i+jlda+2, c2);
                _mm_storeu_pd(C+i+jlda+4, c3);
                _mm_storeu_pd(C+i+jlda+6, c4);
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
void square_dgemm (int lda, double* A, double* B, double* restrict C) {
    /* For each block-row of A */ 
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
    /* For each block-column of B */
        for (int k = 0; k < lda; k += BLOCK_SIZE) {
        /* Accumulate block dgemms into block of C */
            for (int i = 0; i < lda; i += BLOCK_SIZE)   {
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