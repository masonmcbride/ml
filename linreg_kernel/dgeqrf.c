#include <stdbool.h>

static int c__1 = 1;
static double c_b4 = 1.;
static double c_b5 = 0.;
static double c_b14 = 1.;
static double c_b25 = -1.;
static double c_b8 = 0.;

int dlarfb_(char *side, char *trans, char *direct, char *
	storev, int *m, int *n, int *k, double *v, int *
	ldv, double *t, int *ldt, double *c__, int *ldc, 
	double *work, int *ldwork)
{
    /* System generated locals */
    int c_dim1, c_offset, t_dim1, t_offset, v_dim1, v_offset, work_dim1, 
	    work_offset, i__1, i__2;

    /* Local variables */
    int i__, j;
    int lastc;
    int lastv;
    char transt[1];


/*     Quick return if possible */

    /* Parameter adjustments */
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    work_dim1 = *ldwork;
    work_offset = 1 + work_dim1;
    work -= work_offset;

    /* Function Body */
    if (*m <= 0 || *n <= 0) {
	return 0;
    }

    if (lsame_(trans, "N")) {
	*(unsigned char *)transt = 'T';
    } else {
	*(unsigned char *)transt = 'N';
    }

    if (lsame_(storev, "C")) {

	if (lsame_(direct, "F")) {

/*           Let  V =  ( V1 )    (first K rows) */
/*                     ( V2 ) */
/*           where  V1  is unit lower triangular. */

	    if (lsame_(side, "L")) {

/*              Form  H * C  or  H' * C  where  C = ( C1 ) */
/*                                                  ( C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = iladlr_(m, k, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = iladlc_(&lastv, n, &c__[c_offset], ldc);

/*              W := C' * V  =  (C1'*V1 + C2'*V2)  (stored in WORK) */

/*              W := C1' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    dcopy_(&lastc, &c__[j + c_dim1], ldc, &work[j * work_dim1 
			    + 1], &c__1);
/* L10: */
		}

/*              W := W * V1 */

		dtrmm_("Right", "Lower", "No transpose", "Unit", &lastc, k, &
			c_b14, &v[v_offset], ldv, &work[work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C2'*V2 */

		    i__1 = lastv - *k;
		    dgemm_("Transpose", "No transpose", &lastc, k, &i__1, &
			    c_b14, &c__[*k + 1 + c_dim1], ldc, &v[*k + 1 + 
			    v_dim1], ldv, &c_b14, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		dtrmm_("Right", "Upper", transt, "Non-unit", &lastc, k, &
			c_b14, &t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V * W' */

		if (lastv > *k) {

/*                 C2 := C2 - V2 * W' */

		    i__1 = lastv - *k;
		    dgemm_("No transpose", "Transpose", &i__1, &lastc, k, &
			    c_b25, &v[*k + 1 + v_dim1], ldv, &work[
			    work_offset], ldwork, &c_b14, &c__[*k + 1 + 
			    c_dim1], ldc);
		}

/*              W := W * V1' */

		dtrmm_("Right", "Lower", "Transpose", "Unit", &lastc, k, &
			c_b14, &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[j + i__ * c_dim1] -= work[i__ + j * work_dim1];
/* L20: */
		    }
/* L30: */
		}

	    } else if (lsame_(side, "R")) {

/*              Form  C * H  or  C * H'  where  C = ( C1  C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = iladlr_(n, k, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = iladlr_(m, &lastv, &c__[c_offset], ldc);

/*              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK) */

/*              W := C1 */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    dcopy_(&lastc, &c__[j * c_dim1 + 1], &c__1, &work[j * 
			    work_dim1 + 1], &c__1);
/* L40: */
		}

/*              W := W * V1 */

		dtrmm_("Right", "Lower", "No transpose", "Unit", &lastc, k, &
			c_b14, &v[v_offset], ldv, &work[work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C2 * V2 */

		    i__1 = lastv - *k;
		    dgemm_("No transpose", "No transpose", &lastc, k, &i__1, &
			    c_b14, &c__[(*k + 1) * c_dim1 + 1], ldc, &v[*k + 
			    1 + v_dim1], ldv, &c_b14, &work[work_offset], 
			    ldwork);
		}

/*              W := W * T  or  W * T' */

		dtrmm_("Right", "Upper", trans, "Non-unit", &lastc, k, &c_b14, 
			 &t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V' */

		if (lastv > *k) {

/*                 C2 := C2 - W * V2' */

		    i__1 = lastv - *k;
		    dgemm_("No transpose", "Transpose", &lastc, &i__1, k, &
			    c_b25, &work[work_offset], ldwork, &v[*k + 1 + 
			    v_dim1], ldv, &c_b14, &c__[(*k + 1) * c_dim1 + 1], 
			     ldc);
		}

/*              W := W * V1' */

		dtrmm_("Right", "Lower", "Transpose", "Unit", &lastc, k, &
			c_b14, &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] -= work[i__ + j * work_dim1];
/* L50: */
		    }
/* L60: */
		}
	    }

	} else {

/*           Let  V =  ( V1 ) */
/*                     ( V2 )    (last K rows) */
/*           where  V2  is unit upper triangular. */

	    if (lsame_(side, "L")) {

/*              Form  H * C  or  H' * C  where  C = ( C1 ) */
/*                                                  ( C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = iladlr_(m, k, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = iladlc_(&lastv, n, &c__[c_offset], ldc);

/*              W := C' * V  =  (C1'*V1 + C2'*V2)  (stored in WORK) */

/*              W := C2' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    dcopy_(&lastc, &c__[lastv - *k + j + c_dim1], ldc, &work[
			    j * work_dim1 + 1], &c__1);
/* L70: */
		}

/*              W := W * V2 */

		dtrmm_("Right", "Upper", "No transpose", "Unit", &lastc, k, &
			c_b14, &v[lastv - *k + 1 + v_dim1], ldv, &work[
			work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C1'*V1 */

		    i__1 = lastv - *k;
		    dgemm_("Transpose", "No transpose", &lastc, k, &i__1, &
			    c_b14, &c__[c_offset], ldc, &v[v_offset], ldv, &
			    c_b14, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		dtrmm_("Right", "Lower", transt, "Non-unit", &lastc, k, &
			c_b14, &t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V * W' */

		if (lastv > *k) {

/*                 C1 := C1 - V1 * W' */

		    i__1 = lastv - *k;
		    dgemm_("No transpose", "Transpose", &i__1, &lastc, k, &
			    c_b25, &v[v_offset], ldv, &work[work_offset], 
			    ldwork, &c_b14, &c__[c_offset], ldc);
		}

/*              W := W * V2' */

		dtrmm_("Right", "Upper", "Transpose", "Unit", &lastc, k, &
			c_b14, &v[lastv - *k + 1 + v_dim1], ldv, &work[
			work_offset], ldwork);

/*              C2 := C2 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[lastv - *k + j + i__ * c_dim1] -= work[i__ + j * 
				work_dim1];
/* L80: */
		    }
/* L90: */
		}

	    } else if (lsame_(side, "R")) {

/*              Form  C * H  or  C * H'  where  C = ( C1  C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = iladlr_(n, k, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = iladlr_(m, &lastv, &c__[c_offset], ldc);

/*              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK) */

/*              W := C2 */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    dcopy_(&lastc, &c__[(*n - *k + j) * c_dim1 + 1], &c__1, &
			    work[j * work_dim1 + 1], &c__1);
/* L100: */
		}

/*              W := W * V2 */

		dtrmm_("Right", "Upper", "No transpose", "Unit", &lastc, k, &
			c_b14, &v[lastv - *k + 1 + v_dim1], ldv, &work[
			work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C1 * V1 */

		    i__1 = lastv - *k;
		    dgemm_("No transpose", "No transpose", &lastc, k, &i__1, &
			    c_b14, &c__[c_offset], ldc, &v[v_offset], ldv, &
			    c_b14, &work[work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		dtrmm_("Right", "Lower", trans, "Non-unit", &lastc, k, &c_b14, 
			 &t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V' */

		if (lastv > *k) {

/*                 C1 := C1 - W * V1' */

		    i__1 = lastv - *k;
		    dgemm_("No transpose", "Transpose", &lastc, &i__1, k, &
			    c_b25, &work[work_offset], ldwork, &v[v_offset], 
			    ldv, &c_b14, &c__[c_offset], ldc);
		}

/*              W := W * V2' */

		dtrmm_("Right", "Upper", "Transpose", "Unit", &lastc, k, &
			c_b14, &v[lastv - *k + 1 + v_dim1], ldv, &work[
			work_offset], ldwork);

/*              C2 := C2 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + (lastv - *k + j) * c_dim1] -= work[i__ + j *
				 work_dim1];
/* L110: */
		    }
/* L120: */
		}
	    }
	}

    } else if (lsame_(storev, "R")) {

	if (lsame_(direct, "F")) {

/*           Let  V =  ( V1  V2 )    (V1: first K columns) */
/*           where  V1  is unit upper triangular. */

	    if (lsame_(side, "L")) {

/*              Form  H * C  or  H' * C  where  C = ( C1 ) */
/*                                                  ( C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = iladlc_(k, m, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = iladlc_(&lastv, n, &c__[c_offset], ldc);

/*              W := C' * V'  =  (C1'*V1' + C2'*V2') (stored in WORK) */

/*              W := C1' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    dcopy_(&lastc, &c__[j + c_dim1], ldc, &work[j * work_dim1 
			    + 1], &c__1);
/* L130: */
		}

/*              W := W * V1' */

		dtrmm_("Right", "Upper", "Transpose", "Unit", &lastc, k, &
			c_b14, &v[v_offset], ldv, &work[work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C2'*V2' */

		    i__1 = lastv - *k;
		    dgemm_("Transpose", "Transpose", &lastc, k, &i__1, &c_b14, 
			     &c__[*k + 1 + c_dim1], ldc, &v[(*k + 1) * v_dim1 
			    + 1], ldv, &c_b14, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		dtrmm_("Right", "Upper", transt, "Non-unit", &lastc, k, &
			c_b14, &t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V' * W' */

		if (lastv > *k) {

/*                 C2 := C2 - V2' * W' */

		    i__1 = lastv - *k;
		    dgemm_("Transpose", "Transpose", &i__1, &lastc, k, &c_b25, 
			     &v[(*k + 1) * v_dim1 + 1], ldv, &work[
			    work_offset], ldwork, &c_b14, &c__[*k + 1 + 
			    c_dim1], ldc);
		}

/*              W := W * V1 */

		dtrmm_("Right", "Upper", "No transpose", "Unit", &lastc, k, &
			c_b14, &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[j + i__ * c_dim1] -= work[i__ + j * work_dim1];
/* L140: */
		    }
/* L150: */
		}

	    } else if (lsame_(side, "R")) {

/*              Form  C * H  or  C * H'  where  C = ( C1  C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = iladlc_(k, n, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = iladlr_(m, &lastv, &c__[c_offset], ldc);

/*              W := C * V'  =  (C1*V1' + C2*V2')  (stored in WORK) */

/*              W := C1 */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    dcopy_(&lastc, &c__[j * c_dim1 + 1], &c__1, &work[j * 
			    work_dim1 + 1], &c__1);
/* L160: */
		}

/*              W := W * V1' */

		dtrmm_("Right", "Upper", "Transpose", "Unit", &lastc, k, &
			c_b14, &v[v_offset], ldv, &work[work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C2 * V2' */

		    i__1 = lastv - *k;
		    dgemm_("No transpose", "Transpose", &lastc, k, &i__1, &
			    c_b14, &c__[(*k + 1) * c_dim1 + 1], ldc, &v[(*k + 
			    1) * v_dim1 + 1], ldv, &c_b14, &work[work_offset], 
			     ldwork);
		}

/*              W := W * T  or  W * T' */

		dtrmm_("Right", "Upper", trans, "Non-unit", &lastc, k, &c_b14, 
			 &t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V */

		if (lastv > *k) {

/*                 C2 := C2 - W * V2 */

		    i__1 = lastv - *k;
		    dgemm_("No transpose", "No transpose", &lastc, &i__1, k, &
			    c_b25, &work[work_offset], ldwork, &v[(*k + 1) * 
			    v_dim1 + 1], ldv, &c_b14, &c__[(*k + 1) * c_dim1 
			    + 1], ldc);
		}

/*              W := W * V1 */

		dtrmm_("Right", "Upper", "No transpose", "Unit", &lastc, k, &
			c_b14, &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] -= work[i__ + j * work_dim1];
/* L170: */
		    }
/* L180: */
		}

	    }

	} else {

/*           Let  V =  ( V1  V2 )    (V2: last K columns) */
/*           where  V2  is unit lower triangular. */

	    if (lsame_(side, "L")) {

/*              Form  H * C  or  H' * C  where  C = ( C1 ) */
/*                                                  ( C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = iladlc_(k, m, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = iladlc_(&lastv, n, &c__[c_offset], ldc);

/*              W := C' * V'  =  (C1'*V1' + C2'*V2') (stored in WORK) */

/*              W := C2' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    dcopy_(&lastc, &c__[lastv - *k + j + c_dim1], ldc, &work[
			    j * work_dim1 + 1], &c__1);
/* L190: */
		}

/*              W := W * V2' */

		dtrmm_("Right", "Lower", "Transpose", "Unit", &lastc, k, &
			c_b14, &v[(lastv - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C1'*V1' */

		    i__1 = lastv - *k;
		    dgemm_("Transpose", "Transpose", &lastc, k, &i__1, &c_b14, 
			     &c__[c_offset], ldc, &v[v_offset], ldv, &c_b14, &
			    work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		dtrmm_("Right", "Lower", transt, "Non-unit", &lastc, k, &
			c_b14, &t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V' * W' */

		if (lastv > *k) {

/*                 C1 := C1 - V1' * W' */

		    i__1 = lastv - *k;
		    dgemm_("Transpose", "Transpose", &i__1, &lastc, k, &c_b25, 
			     &v[v_offset], ldv, &work[work_offset], ldwork, &
			    c_b14, &c__[c_offset], ldc);
		}

/*              W := W * V2 */

		dtrmm_("Right", "Lower", "No transpose", "Unit", &lastc, k, &
			c_b14, &v[(lastv - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);

/*              C2 := C2 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[lastv - *k + j + i__ * c_dim1] -= work[i__ + j * 
				work_dim1];
/* L200: */
		    }
/* L210: */
		}

	    } else if (lsame_(side, "R")) {

/*              Form  C * H  or  C * H'  where  C = ( C1  C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = iladlc_(k, n, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = iladlr_(m, &lastv, &c__[c_offset], ldc);

/*              W := C * V'  =  (C1*V1' + C2*V2')  (stored in WORK) */

/*              W := C2 */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    dcopy_(&lastc, &c__[(lastv - *k + j) * c_dim1 + 1], &c__1, 
			     &work[j * work_dim1 + 1], &c__1);
/* L220: */
		}

/*              W := W * V2' */

		dtrmm_("Right", "Lower", "Transpose", "Unit", &lastc, k, &
			c_b14, &v[(lastv - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C1 * V1' */

		    i__1 = lastv - *k;
		    dgemm_("No transpose", "Transpose", &lastc, k, &i__1, &
			    c_b14, &c__[c_offset], ldc, &v[v_offset], ldv, &
			    c_b14, &work[work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		dtrmm_("Right", "Lower", trans, "Non-unit", &lastc, k, &c_b14, 
			 &t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V */

		if (lastv > *k) {

/*                 C1 := C1 - W * V1 */

		    i__1 = lastv - *k;
		    dgemm_("No transpose", "No transpose", &lastc, &i__1, k, &
			    c_b25, &work[work_offset], ldwork, &v[v_offset], 
			    ldv, &c_b14, &c__[c_offset], ldc);
		}

/*              W := W * V2 */

		dtrmm_("Right", "Lower", "No transpose", "Unit", &lastc, k, &
			c_b14, &v[(lastv - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + (lastv - *k + j) * c_dim1] -= work[i__ + j *
				 work_dim1];
/* L230: */
		    }
/* L240: */
		}

	    }

	}
    }

    return 0;

/*     End of DLARFB */

} /* dlarfb_ */


/* Subroutine */ int dgeqr2_(int *m, int *n, double *a, int *
	lda, double *tau, double *work, int *info)
{
    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    int i__, k;
    double aii;

/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGEQR2 computes a QR factorization of a real m by n matrix A: */
/*  A = Q * R. */

/*  Arguments */
/*  ========= */

/*  M       (input) int */
/*          The number of rows of the matrix A.  M >= 0. */

/*  N       (input) int */
/*          The number of columns of the matrix A.  N >= 0. */

/*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N) */
/*          On entry, the m by n matrix A. */
/*          On exit, the elements on and above the diagonal of the array */
/*          contain the min(m,n) by n upper trapezoidal matrix R (R is */
/*          upper triangular if m >= n); the elements below the diagonal, */
/*          with the array TAU, represent the orthogonal matrix Q as a */
/*          product of elementary reflectors (see Further Details). */

/*  LDA     (input) int */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  TAU     (output) DOUBLE PRECISION array, dimension (min(M,N)) */
/*          The scalar factors of the elementary reflectors (see Further */
/*          Details). */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension (N) */

/*  INFO    (output) int */
/*          = 0: successful exit */
/*          < 0: if INFO = -i, the i-th argument had an illegal value */

/*  Further Details */
/*  =============== */

/*  The matrix Q is represented as a product of elementary reflectors */

/*     Q = H(1) H(2) . . . H(k), where k = min(m,n). */

/*  Each H(i) has the form */

/*     H(i) = I - tau * v * v' */

/*  where tau is a real scalar, and v is a real vector with */
/*  v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i), */
/*  and tau in TAU(i). */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGEQR2", &i__1);
	return 0;
    }

    k = min(*m,*n);

    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Generate elementary reflector H(i) to annihilate A(i+1:m,i) */

	i__2 = *m - i__ + 1;
/* Computing MIN */
	i__3 = i__ + 1;
	dlarfp_(&i__2, &a[i__ + i__ * a_dim1], &a[min(i__3, *m)+ i__ * a_dim1]
, &c__1, &tau[i__]);
	if (i__ < *n) {

/*           Apply H(i) to A(i:m,i+1:n) from the left */

	    aii = a[i__ + i__ * a_dim1];
	    a[i__ + i__ * a_dim1] = 1.;
	    i__2 = *m - i__ + 1;
	    i__3 = *n - i__;
	    dlarf_("Left", &i__2, &i__3, &a[i__ + i__ * a_dim1], &c__1, &tau[
		    i__], &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	    a[i__ + i__ * a_dim1] = aii;
	}
/* L10: */
    }
    return 0;

/*     End of DGEQR2 */

} /* dgeqr2_ */

int dlarft_(char *direct, char *storev, int *n, int *k, double *v, int *ldv, double *tau, double *t, int *ldt)
{
    /* System generated locals */
    int t_dim1, t_offset, v_dim1, v_offset, i__1, i__2, i__3;
    double d__1;

    /* Local variables */
    int i__, j, prevlastv;
    double vii;
    int lastv;

/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLARFT forms the triangular factor T of a real block reflector H */
/*  of order n, which is defined as a product of k elementary reflectors. */

/*  If DIRECT = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular; */

/*  If DIRECT = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular. */

/*  If STOREV = 'C', the vector which defines the elementary reflector */
/*  H(i) is stored in the i-th column of the array V, and */

/*     H  =  I - V * T * V' */

/*  If STOREV = 'R', the vector which defines the elementary reflector */
/*  H(i) is stored in the i-th row of the array V, and */

/*     H  =  I - V' * T * V */

/*  Arguments */
/*  ========= */

/*  DIRECT  (input) CHARACTER*1 */
/*          Specifies the order in which the elementary reflectors are */
/*          multiplied to form the block reflector: */
/*          = 'F': H = H(1) H(2) . . . H(k) (Forward) */
/*          = 'B': H = H(k) . . . H(2) H(1) (Backward) */

/*  STOREV  (input) CHARACTER*1 */
/*          Specifies how the vectors which define the elementary */
/*          reflectors are stored (see also Further Details): */
/*          = 'C': columnwise */
/*          = 'R': rowwise */

/*  N       (input) int */
/*          The order of the block reflector H. N >= 0. */

/*  K       (input) int */
/*          The order of the triangular factor T (= the number of */
/*          elementary reflectors). K >= 1. */

/*  V       (input/output) DOUBLE PRECISION array, dimension */
/*                               (LDV,K) if STOREV = 'C' */
/*                               (LDV,N) if STOREV = 'R' */
/*          The matrix V. See further details. */

/*  LDV     (input) int */
/*          The leading dimension of the array V. */
/*          If STOREV = 'C', LDV >= max(1,N); if STOREV = 'R', LDV >= K. */

/*  TAU     (input) DOUBLE PRECISION array, dimension (K) */
/*          TAU(i) must contain the scalar factor of the elementary */
/*          reflector H(i). */

/*  T       (output) DOUBLE PRECISION array, dimension (LDT,K) */
/*          The k by k triangular factor T of the block reflector. */
/*          If DIRECT = 'F', T is upper triangular; if DIRECT = 'B', T is */
/*          lower triangular. The rest of the array is not used. */

/*  LDT     (input) int */
/*          The leading dimension of the array T. LDT >= K. */

/*  Further Details */
/*  =============== */

/*  The shape of the matrix V and the storage of the vectors which define */
/*  the H(i) is best illustrated by the following example with n = 5 and */
/*  k = 3. The elements equal to 1 are not stored; the corresponding */
/*  array elements are modified but restored on exit. The rest of the */
/*  array is not used. */

/*  DIRECT = 'F' and STOREV = 'C':         DIRECT = 'F' and STOREV = 'R': */

/*               V = (  1       )                 V = (  1 v1 v1 v1 v1 ) */
/*                   ( v1  1    )                     (     1 v2 v2 v2 ) */
/*                   ( v1 v2  1 )                     (        1 v3 v3 ) */
/*                   ( v1 v2 v3 ) */
/*                   ( v1 v2 v3 ) */

/*  DIRECT = 'B' and STOREV = 'C':         DIRECT = 'B' and STOREV = 'R': */

/*               V = ( v1 v2 v3 )                 V = ( v1 v1  1       ) */
/*                   ( v1 v2 v3 )                     ( v2 v2 v2  1    ) */
/*                   (  1 v2 v3 )                     ( v3 v3 v3 v3  1 ) */
/*                   (     1 v3 ) */
/*                   (        1 ) */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Quick return if possible */

    /* Parameter adjustments */
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    --tau;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;

    /* Function Body */
    if (*n == 0) {
	return 0;
    }

    if (lsame_(direct, "F")) {
	prevlastv = *n;
	i__1 = *k;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    prevlastv = max(i__,prevlastv);
	    if (tau[i__] == 0.) {

/*              H(i)  =  I */

		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    t[j + i__ * t_dim1] = 0.;
/* L10: */
		}
	    } else {

/*              general case */

		vii = v[i__ + i__ * v_dim1];
		v[i__ + i__ * v_dim1] = 1.;
		if (lsame_(storev, "C")) {
/*                 Skip any trailing zeros. */
		    i__2 = i__ + 1;
		    for (lastv = *n; lastv >= i__2; --lastv) {
			if (v[lastv + i__ * v_dim1] != 0.) {
			    break;
			}
		    }
		    j = min(lastv,prevlastv);

/*                 T(1:i-1,i) := - tau(i) * V(i:j,1:i-1)' * V(i:j,i) */

		    i__2 = j - i__ + 1;
		    i__3 = i__ - 1;
		    d__1 = -tau[i__];
		    dgemv_("Transpose", &i__2, &i__3, &d__1, &v[i__ + v_dim1], 
			     ldv, &v[i__ + i__ * v_dim1], &c__1, &c_b8, &t[
			    i__ * t_dim1 + 1], &c__1);
		} else {
/*                 Skip any trailing zeros. */
		    i__2 = i__ + 1;
		    for (lastv = *n; lastv >= i__2; --lastv) {
			if (v[i__ + lastv * v_dim1] != 0.) {
			    break;
			}
		    }
		    j = min(lastv,prevlastv);

/*                 T(1:i-1,i) := - tau(i) * V(1:i-1,i:j) * V(i,i:j)' */

		    i__2 = i__ - 1;
		    i__3 = j - i__ + 1;
		    d__1 = -tau[i__];
		    dgemv_("No transpose", &i__2, &i__3, &d__1, &v[i__ * 
			    v_dim1 + 1], ldv, &v[i__ + i__ * v_dim1], ldv, &
			    c_b8, &t[i__ * t_dim1 + 1], &c__1);
		}
		v[i__ + i__ * v_dim1] = vii;

/*              T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i) */

		i__2 = i__ - 1;
		dtrmv_("Upper", "No transpose", "Non-unit", &i__2, &t[
			t_offset], ldt, &t[i__ * t_dim1 + 1], &c__1);
		t[i__ + i__ * t_dim1] = tau[i__];
		if (i__ > 1) {
		    prevlastv = max(prevlastv,lastv);
		} else {
		    prevlastv = lastv;
		}
	    }
/* L20: */
	}
    } else {
	prevlastv = 1;
	for (i__ = *k; i__ >= 1; --i__) {
	    if (tau[i__] == 0.) {

/*              H(i)  =  I */

		i__1 = *k;
		for (j = i__; j <= i__1; ++j) {
		    t[j + i__ * t_dim1] = 0.;
/* L30: */
		}
	    } else {

/*              general case */

		if (i__ < *k) {
		    if (lsame_(storev, "C")) {
			vii = v[*n - *k + i__ + i__ * v_dim1];
			v[*n - *k + i__ + i__ * v_dim1] = 1.;
/*                    Skip any leading zeros. */
			i__1 = i__ - 1;
			for (lastv = 1; lastv <= i__1; ++lastv) {
			    if (v[lastv + i__ * v_dim1] != 0.) {
				break;
			    }
			}
			j = max(lastv,prevlastv);

/*                    T(i+1:k,i) := */
/*                            - tau(i) * V(j:n-k+i,i+1:k)' * V(j:n-k+i,i) */

			i__1 = *n - *k + i__ - j + 1;
			i__2 = *k - i__;
			d__1 = -tau[i__];
			dgemv_("Transpose", &i__1, &i__2, &d__1, &v[j + (i__ 
				+ 1) * v_dim1], ldv, &v[j + i__ * v_dim1], &
				c__1, &c_b8, &t[i__ + 1 + i__ * t_dim1], &
				c__1);
			v[*n - *k + i__ + i__ * v_dim1] = vii;
		    } else {
			vii = v[i__ + (*n - *k + i__) * v_dim1];
			v[i__ + (*n - *k + i__) * v_dim1] = 1.;
/*                    Skip any leading zeros. */
			i__1 = i__ - 1;
			for (lastv = 1; lastv <= i__1; ++lastv) {
			    if (v[i__ + lastv * v_dim1] != 0.) {
				break;
			    }
			}
			j = max(lastv,prevlastv);

/*                    T(i+1:k,i) := */
/*                            - tau(i) * V(i+1:k,j:n-k+i) * V(i,j:n-k+i)' */

			i__1 = *k - i__;
			i__2 = *n - *k + i__ - j + 1;
			d__1 = -tau[i__];
			dgemv_("No transpose", &i__1, &i__2, &d__1, &v[i__ + 
				1 + j * v_dim1], ldv, &v[i__ + j * v_dim1], 
				ldv, &c_b8, &t[i__ + 1 + i__ * t_dim1], &c__1);
			v[i__ + (*n - *k + i__) * v_dim1] = vii;
		    }

/*                 T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i) */

		    i__1 = *k - i__;
		    dtrmv_("Lower", "No transpose", "Non-unit", &i__1, &t[i__ 
			    + 1 + (i__ + 1) * t_dim1], ldt, &t[i__ + 1 + i__ *
			     t_dim1], &c__1)
			    ;
		    if (i__ > 1) {
			prevlastv = min(prevlastv,lastv);
		    } else {
			prevlastv = lastv;
		    }
		}
		t[i__ + i__ * t_dim1] = tau[i__];
	    }
/* L40: */
	}
    }
    return 0;

/*     End of DLARFT */

} /* dlarft_ */

/* Subroutine */ int dlarfp_(int *n, double *alpha, double *x, 
	int *incx, double *tau)
{
    /* System generated locals */
    int i__1;
    double d__1;

    /* Builtin functions */
    double d_sign(double *, double *);

    /* Local variables */
    int j, knt;
    double beta;
    double xnorm;
    double safmin, rsafmn;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLARFP generates a real elementary reflector H of order n, such */
/*  that */

/*        H * ( alpha ) = ( beta ),   H' * H = I. */
/*            (   x   )   (   0  ) */

/*  where alpha and beta are scalars, beta is non-negative, and x is */
/*  an (n-1)-element real vector.  H is represented in the form */

/*        H = I - tau * ( 1 ) * ( 1 v' ) , */
/*                      ( v ) */

/*  where tau is a real scalar and v is a real (n-1)-element */
/*  vector. */

/*  If the elements of x are all zero, then tau = 0 and H is taken to be */
/*  the unit matrix. */

/*  Otherwise  1 <= tau <= 2. */

/*  Arguments */
/*  ========= */

/*  N       (input) int */
/*          The order of the elementary reflector. */

/*  ALPHA   (input/output) DOUBLE PRECISION */
/*          On entry, the value alpha. */
/*          On exit, it is overwritten with the value beta. */

/*  X       (input/output) DOUBLE PRECISION array, dimension */
/*                         (1+(N-2)*abs(INCX)) */
/*          On entry, the vector x. */
/*          On exit, it is overwritten with the vector v. */

/*  INCX    (input) int */
/*          The increment between elements of X. INCX > 0. */

/*  TAU     (output) DOUBLE PRECISION */
/*          The value tau. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n <= 0) {
	*tau = 0.;
	return 0;
    }

    i__1 = *n - 1;
    xnorm = dnrm2_(&i__1, &x[1], incx);

    if (xnorm == 0.) {

/*        H  =  [+/-1, 0; I], sign chosen so ALPHA >= 0 */

	if (*alpha >= 0.) {
/*           When TAU.eq.ZERO, the vector is special-cased to be */
/*           all zeros in the application routines.  We do not need */
/*           to clear it. */
	    *tau = 0.;
	} else {
/*           However, the application routines rely on explicit */
/*           zero checks when TAU.ne.ZERO, and we must clear X. */
	    *tau = 2.;
	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		x[(j - 1) * *incx + 1] = 0.;
	    }
	    *alpha = -(*alpha);
	}
    } else {

/*        general case */

	d__1 = dlapy2_(alpha, &xnorm);
	beta = d_sign(&d__1, alpha);
	safmin = dlamch_("S") / dlamch_("E");
	knt = 0;
	if (abs(beta) < safmin) {

/*           XNORM, BETA may be inaccurate; scale X and recompute them */

	    rsafmn = 1. / safmin;
L10:
	    ++knt;
	    i__1 = *n - 1;
	    dscal_(&i__1, &rsafmn, &x[1], incx);
	    beta *= rsafmn;
	    *alpha *= rsafmn;
	    if (abs(beta) < safmin) {
		goto L10;
	    }

/*           New BETA is at most 1, at least SAFMIN */

	    i__1 = *n - 1;
	    xnorm = dnrm2_(&i__1, &x[1], incx);
	    d__1 = dlapy2_(alpha, &xnorm);
	    beta = d_sign(&d__1, alpha);
	}
	*alpha += beta;
	if (beta < 0.) {
	    beta = -beta;
	    *tau = -(*alpha) / beta;
	} else {
	    *alpha = xnorm * (xnorm / *alpha);
	    *tau = *alpha / beta;
	    *alpha = -(*alpha);
	}
	i__1 = *n - 1;
	d__1 = 1. / *alpha;
	dscal_(&i__1, &d__1, &x[1], incx);

/*        If BETA is subnormal, it may lose relative accuracy */

	i__1 = knt;
	for (j = 1; j <= i__1; ++j) {
	    beta *= safmin;
/* L20: */
	}
	*alpha = beta;
    }

    return 0;

/*     End of DLARFP */

} /* dlarfp_ */

/* Subroutine */ int dlarf_(char *side, int *m, int *n, double *v, 
	 int *incv, double *tau, double *c__, int *ldc, 
	double *work)
{
    /* System generated locals */
    int c_dim1, c_offset;
    double d__1;

    /* Local variables */
    int i__;
    bool applyleft;
    int lastc, lastv;

/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLARF applies a real elementary reflector H to a real m by n matrix */
/*  C, from either the left or the right. H is represented in the form */

/*        H = I - tau * v * v' */

/*  where tau is a real scalar and v is a real vector. */

/*  If tau = 0, then H is taken to be the unit matrix. */

/*  Arguments */
/*  ========= */

/*  SIDE    (input) CHARACTER*1 */
/*          = 'L': form  H * C */
/*          = 'R': form  C * H */

/*  M       (input) int */
/*          The number of rows of the matrix C. */

/*  N       (input) int */
/*          The number of columns of the matrix C. */

/*  V       (input) DOUBLE PRECISION array, dimension */
/*                     (1 + (M-1)*abs(INCV)) if SIDE = 'L' */
/*                  or (1 + (N-1)*abs(INCV)) if SIDE = 'R' */
/*          The vector v in the representation of H. V is not used if */
/*          TAU = 0. */

/*  INCV    (input) int */
/*          The increment between elements of v. INCV <> 0. */

/*  TAU     (input) DOUBLE PRECISION */
/*          The value tau in the representation of H. */

/*  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N) */
/*          On entry, the m by n matrix C. */
/*          On exit, C is overwritten by the matrix H * C if SIDE = 'L', */
/*          or C * H if SIDE = 'R'. */

/*  LDC     (input) int */
/*          The leading dimension of the array C. LDC >= max(1,M). */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension */
/*                         (N) if SIDE = 'L' */
/*                      or (M) if SIDE = 'R' */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --v;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    applyleft = lsame_(side, "L");
    lastv = 0;
    lastc = 0;
    if (*tau != 0.) {
/*     Set up variables for scanning V.  LASTV begins pointing to the end */
/*     of V. */
	if (applyleft) {
	    lastv = *m;
	} else {
	    lastv = *n;
	}
	if (*incv > 0) {
	    i__ = (lastv - 1) * *incv + 1;
	} else {
	    i__ = 1;
	}
/*     Look for the last non-zero row in V. */
	while(lastv > 0 && v[i__] == 0.) {
	    --lastv;
	    i__ -= *incv;
	}
	if (applyleft) {
/*     Scan for the last non-zero column in C(1:lastv,:). */
	    lastc = iladlc_(&lastv, n, &c__[c_offset], ldc);
	} else {
/*     Scan for the last non-zero row in C(:,1:lastv). */
	    lastc = iladlr_(m, &lastv, &c__[c_offset], ldc);
	}
    }
/*     Note that lastc.eq.0 renders the BLAS operations null; no special */
/*     case is needed at this level. */
    if (applyleft) {

/*        Form  H * C */

	if (lastv > 0) {

/*           w(1:lastc,1) := C(1:lastv,1:lastc)' * v(1:lastv,1) */

	    dgemv_("Transpose", &lastv, &lastc, &c_b4, &c__[c_offset], ldc, &
		    v[1], incv, &c_b5, &work[1], &c__1);

/*           C(1:lastv,1:lastc) := C(...) - v(1:lastv,1) * w(1:lastc,1)' */

	    d__1 = -(*tau);
	    dger_(&lastv, &lastc, &d__1, &v[1], incv, &work[1], &c__1, &c__[
		    c_offset], ldc);
	}
    } else {

/*        Form  C * H */

	if (lastv > 0) {

/*           w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv,1) */

	    dgemv_("No transpose", &lastc, &lastv, &c_b4, &c__[c_offset], ldc, 
		     &v[1], incv, &c_b5, &work[1], &c__1);

/*           C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv,1)' */

	    d__1 = -(*tau);
	    dger_(&lastc, &lastv, &d__1, &work[1], &c__1, &v[1], incv, &c__[
		    c_offset], ldc);
	}
    }
    return 0;

/*     End of DLARF */

} /* dlarf_ */

/* Subroutine */ int dgeqrf_(int *m, int *n, double *a, int *
	lda, double *tau, double *work, int *lwork, int *info)
{
    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    int i__, k, ib, nb, nx, iws, nbmin, iinfo;
    int ldwork, lwkopt;
    bool lquery;

/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    //nb = ilaenv_(&c__1, "DGEQRF", " ", m, n, &c_n1, &c_n1);
    lwkopt = *n * nb;
    work[1] = (double) lwkopt;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	//xerbla_("DGEQRF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    k = min(*m,*n);
    if (k == 0) {
	work[1] = 1.;
	return 0;
    }

    nbmin = 2;
    nx = 0;
    iws = *n;
    if (nb > 1 && nb < k) {

/*        Determine when to cross over from blocked to unblocked code. */

/* Computing MAX */
	//i__1 = 0, i__2 = ilaenv_(&c__3, "DGEQRF", " ", m, n, &c_n1, &c_n1);
	nx = max(i__1,i__2);
	if (nx < k) {

/*           Determine if workspace is large enough for blocked code. */

	    ldwork = *n;
	    iws = ldwork * nb;
	    if (*lwork < iws) {

/*              Not enough workspace to use optimal NB:  reduce NB and */
/*              determine the minimum value of NB. */

		nb = *lwork / ldwork;
/* Computing MAX */
		//i__1 = 2, i__2 = ilaenv_(&c__2, "DGEQRF", " ", m, n, &c_n1, &c_n1);
		nbmin = max(i__1,i__2);
	    }
	}
    }

    if (nb >= nbmin && nb < k && nx < k) {

/*        Use blocked code initially */

	i__1 = k - nx;
	i__2 = nb;
	for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
	    i__3 = k - i__ + 1;
	    ib = min(i__3,nb);

/*           Compute the QR factorization of the current block */
/*           A(i:m,i:i+ib-1) */

	    i__3 = *m - i__ + 1;
	    dgeqr2_(&i__3, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[1], &iinfo);
	    if (i__ + ib <= *n) {

/*              Form the triangular factor of the block reflector */
/*              H = H(i) H(i+1) . . . H(i+ib-1) */

		i__3 = *m - i__ + 1;
		dlarft_("Forward", "Columnwise", &i__3, &ib, &a[i__ + i__ * a_dim1], 
                lda, &tau[i__], &work[1], &ldwork);

/*              Apply H' to A(i:m,i+ib:n) from the left */

		i__3 = *m - i__ + 1;
		i__4 = *n - i__ - ib + 1;
		dlarfb_("Left", "Transpose", "Forward", "Columnwise", &i__3, &i__4, 
                &ib, &a[i__ + i__ * a_dim1], lda, &work[1], &ldwork, 
                &a[i__ + (i__ + ib) * a_dim1], lda, &work[ib + 1], &ldwork);
	    }
/* L10: */
	}
    } else {
	i__ = 1;
    }

/*     Use unblocked code to factor the last or only block. */

    if (i__ <= k) {
	i__2 = *m - i__ + 1;
	i__1 = *n - i__ + 1;
	dgeqr2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[1], &iinfo);
    }

    work[1] = (double) iws;
    return 0;

/*     End of DGEQRF */

} /* dgeqrf_ */
