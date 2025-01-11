#include <stdio.h> 
#include "blis.h"

void dgeqp3(int m, int n, double** a, int lda, int* jpvt, 
            double* tau, double* work, int lwork, int info) {
/*  DGEQP3 computes a QR factorization with column pivoting of a
    matrix A:  A*P = Q*R  using Level 3 BLAS. */

/* Do Column Pivoting */
    int nfxd = 0;
    for (int j = 0; j < n; j++) {
        if (jpvt[j] != 0) {
            if (j != nfxd) {
                bli_dswapv(m, a[1,j],1, a[1,nfxd],1);
                jpvt[j] = jpvt[nfxd];
                jpvt[nfxd] = j;
            } else {
                jpvt[j] = j;
            }
            nfxd += 1;
        } else {
            jpvt[j] = j;
        }
    }
    nfxd -= 1;

/* Factorize fixed columns
* Compute the QR factorization of fixed columns
* and update remaining columns
*/
    if (nfxd > 0) {
        int na = min( m, nfxd );
        dgeqrf( m, na, a, lda, tau, work, lwork, info);
        int iws = max( iws, (int)work[1] );
        if (na < n) {
            dormqr('LEFT', 'Transpose', m, n-na, na, a, lda, tau,
                    a[1,na+1], lda, work, lwork, info );
            iws = max( iws, (int)work[1] );
        }
    }

/* Factorize free columns */

    int minmn = min( m, n );
    if (nfxd < minmn) {
        int sm = m - nfxd;
        int sn = n - nfxd;
        int sminmn = minmn - nfxd;

        // Determine the block size
        int inb = 1, inbmin = 2, ixover = 3;
        int nb = ilaev( inb, 'DGEQRF', ' ', sm, sn, -1, -1 );
        int nbmin = 2;
        int nx = 0;

        if (nb > 1 && nb < sminmn) {
            // Determine when to cross over from blocked to unblocked code
            nx = max( 0, ilaenv(ixover, 'DGEQRF', ' ', sm, sn, -1, -1 ));

            if ( nx < sminmn ) {
                // Determine if workspace is large enough for blocked code
                int minws = 2*sn + ( sn+1 )*nb;
                iws = max( iws, minws );
                if ( lwork < minws ) {
                    // Not enough workspace to use optimal NB: Reduce NB
                    // and determine the minimum vlaue of NB. 
                    nb = ( lwork-2*sn ) / ( sn+1 );
                    nbmin = max( 2, ilaenv( inbmin, 'DGEQRF', ' ', sm, sn, -1, -1) );
                }
            }
        }

/* Initialize partial column norms. The first N elements of work store the exact column norms. */
    for (int j = nfxd; j < n; j++) {
        double norm = bli_dnrm2v( sm, a[nfxd+1,j], 1 ); 
        work[n+j] = work[j];
    }  

    if ( nb >= nbmin && nb < sminmn && nx < sminmn ) {
        int j = nfxd + 1;
        int topbmn = minmn - nx;
        int jb;
        while ( j <= topbmn ) {
            jb = min( nb, topbmn-j+1 );

            // Factorize jb columns amoung columns j:n
            int fjb;
            dlaqps( m, n-j+1, j-1, jb, fjb, a[1,j], lda, 
                    jpvt[j], tau[j], work[j], work[n+j],
                    work[2*n+1], work[2*n+jb+1], n-j+1 );

            j += fjb;
        }
    } else {
        int j = nfxd + 1;
    }

    if ( j <= minmn ) {
        dlaqp2( m, n-j+1, j-1, a[1,j], lda, jpvt[j], 
                tau[j], work[j], work[n+j], work[2*n+1]);
    }

    work[1] = iws;
    }

}

int main( int argc, char** argv ) {
    return 0;
}

