#include <stdio.h> 
#include "blis.h"

#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) <= (b) ? (b) : (a))

static int c__1 = 1;
static int c_n1 = -1;
static int c__3 = 3;
static int c__2 = 2;

int dgeqrf_(int *m, int *n, double *a, int *lda, 
			double *tau, double *work, int *lwork, int *info);

int dgeqp3(int *m, int *n, double *a, int *
	lda, int *jpvt, double *tau, double *work, int *lwork, 
	int *info)
{
    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2, i__3;

    int j, jb, na, nb, sm, sn, nx, fjb, iws, nfxd;
    int nbmin, minmn;
    int minws;
    int topbmn, sminmn;
    int lwkopt;
    bool lquery;

/*     Test input arguments */
/*     ==================== */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --jpvt;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }

    if (*info == 0) {
	minmn = min(*m,*n);
	if (minmn == 0) {
	    iws = 1;
	    lwkopt = 1;
	} else {
	    iws = *n * 3 + 1;
	    //nb = ilaenv(&c__1, "DGEQRF", " ", m, n, &c_n1, &c_n1);
	    lwkopt = (*n << 1) + (*n + 1) * nb;
	}
	work[1] = (double) lwkopt;

	if (*lwork < iws && ! lquery) {
	    *info = -8;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	//xerbla_("DGEQP3", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible. */

    if (minmn == 0) {
	return 0;
    }

/*     Move initial columns up front. */

    nfxd = 1;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (jpvt[j] != 0) {
	    if (j != nfxd) {
		dswap_(m, &a[j * a_dim1 + 1], &c__1, &a[nfxd * a_dim1 + 1], &
			c__1);
		jpvt[j] = jpvt[nfxd];
		jpvt[nfxd] = j;
	    } else {
		jpvt[j] = j;
	    }
	    ++nfxd;
	} else {
	    jpvt[j] = j;
	}
/* L10: */
    }
    --nfxd;

/*     Factorize fixed columns */
/*     ======================= */

/*     Compute the QR factorization of fixed columns and update */
/*     remaining columns. */

    if (nfxd > 0) {
	na = min(*m,nfxd);
/* CC      CALL DGEQR2( M, NA, A, LDA, TAU, WORK, INFO ) */
	dgeqrf_(m, &na, &a[a_offset], lda, &tau[1], &work[1], lwork, info);
/* Computing MAX */
	i__1 = iws, i__2 = (int) work[1];
	iws = max(i__1,i__2);
	if (na < *n) {
/* CC         CALL DORM2R( 'Left', 'Transpose', M, N-NA, NA, A, LDA, */
/* CC  $                   TAU, A( 1, NA+1 ), LDA, WORK, INFO ) */
	    i__1 = *n - na;
	    dormqr_("Left", "Transpose", m, &i__1, &na, &a[a_offset], lda, &
		    tau[1], &a[(na + 1) * a_dim1 + 1], lda, &work[1], lwork, 
		    info);
/* Computing MAX */
	    i__1 = iws, i__2 = (int) work[1];
	    iws = max(i__1,i__2);
	}
    }

/*     Factorize free columns */
/*     ====================== */

    if (nfxd < minmn) {

	sm = *m - nfxd;
	sn = *n - nfxd;
	sminmn = minmn - nfxd;

/*        Determine the block size. */

	//nb = ilaenv(&c__1, "DGEQRF", " ", &sm, &sn, &c_n1, &c_n1);
	nbmin = 2;
	nx = 0;

	if (nb > 1 && nb < sminmn) {

/*           Determine when to cross over from blocked to unblocked code. */

/* Computing MAX */
	    //i__1 = 0, i__2 = ilaenv(&c__3, "DGEQRF", " ", &sm, &sn, &c_n1, &c_n1);
	    nx = max(i__1,i__2);


	    if (nx < sminmn) {

/*              Determine if workspace is large enough for blocked code. */

		minws = (sn << 1) + (sn + 1) * nb;
		iws = max(iws,minws);
		if (*lwork < minws) {

/*                 Not enough workspace to use optimal NB: Reduce NB and */
/*                 determine the minimum value of NB. */

		    nb = (*lwork - (sn << 1)) / (sn + 1);
/* Computing MAX */
		    //i__1 = 2, i__2 = ilaenv(&c__2, "DGEQRF", " ", &sm, &sn, &c_n1, &c_n1);
		    nbmin = max(i__1,i__2);


		}
	    }
	}

/*        Initialize partial column norms. The first N elements of work */
/*        store the exact column norms. */

	i__1 = *n;
	for (j = nfxd + 1; j <= i__1; ++j) {
	    work[j] = dnrm2_(&sm, &a[nfxd + 1 + j * a_dim1], &c__1);
	    work[*n + j] = work[j];
/* L20: */
	}

	if (nb >= nbmin && nb < sminmn && nx < sminmn) {

/*           Use blocked code initially. */

	    j = nfxd + 1;

/*           Compute factorization: while loop. */


	    topbmn = minmn - nx;
L30:
	    if (j <= topbmn) {
/* Computing MIN */
		i__1 = nb, i__2 = topbmn - j + 1;
		jb = min(i__1,i__2);

/*              Factorize JB columns among columns J:N. */

		i__1 = *n - j + 1;
		i__2 = j - 1;
		i__3 = *n - j + 1;
		dlaqps_(m, &i__1, &i__2, &jb, &fjb, &a[j * a_dim1 + 1], lda, 
                &jpvt[j], &tau[j], &work[j], &work[*n + j], &work[(*n << 1) + 1],
                &work[(*n << 1) + jb + 1], &i__3);

		j += fjb;
		goto L30;
	    }
	} else {
	    j = nfxd + 1;
	}

/*        Use unblocked code to factor the last or only block. */


	if (j <= minmn) {
	    i__1 = *n - j + 1;
	    i__2 = j - 1;
	    dlaqp2_(m, &i__1, &i__2, &a[j * a_dim1 + 1], lda, &jpvt[j], 
                &tau[j], &work[j], &work[*n + j], &work[(*n << 1) + 1]);
	}

    }

    work[1] = (double) iws;
    return 0;

/*     End of DGEQP3 */

} /* dgeqp3_ */

