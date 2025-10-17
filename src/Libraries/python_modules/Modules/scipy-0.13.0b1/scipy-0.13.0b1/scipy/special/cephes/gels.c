/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "protos.h"

int gels(A, R, M, EPS, AUX)
double A[], R[];
int M;
double EPS;
double AUX[];
{
    int I = 0, J = 0, K, L, IER;
    int II, LL, LLD, LR, LT, LST, LLST, LEND;
    double tb, piv, tol, pivi;

    if (M <= 0) {
      fatal:
	IER = -1;
	goto done;
    }
    /* SEARCH FOR GREATEST MAIN DIAGONAL ELEMENT */

    /*  Diagonal elements are at A(i,i) = 1, 3, 6, 10, ...
     *  A(i,j) = A( i(i-1)/2 + j )
     */
    IER = 0;
    piv = 0.0;
    L = 0;
    for (K = 1; K <= M; K++) {
	L += K;
	tb = fabs(A[L - 1]);
	if (tb > piv) {
	    piv = tb;
	    I = L;
	    J = K;
	}
    }
    tol = EPS * piv;

    /*
     * C     MAIN DIAGONAL ELEMENT A(I)=A(J,J) IS FIRST PIVOT ELEMENT.
     * C     PIV CONTAINS THE ABSOLUTE VALUE OF A(I).
     */

    /*     START ELIMINATION LOOP */
    LST = 0;
    LEND = M - 1;
    for (K = 1; K <= M; K++) {
	/*     TEST ON USEFULNESS OF SYMMETRIC ALGORITHM */
	if (piv <= 0.0)
	    goto fatal;
	if (IER == 0) {
	    if (piv <= tol) {
		IER = K - 1;
	    }
	}
	LT = J - K;
	LST += K;

	/*  PIVOT ROW REDUCTION AND ROW INTERCHANGE IN RIGHT HAND SIDE R */
	pivi = 1.0 / A[I - 1];
	L = K;
	LL = L + LT;
	tb = pivi * R[LL - 1];
	R[LL - 1] = R[L - 1];
	R[L - 1] = tb;
	/* IS ELIMINATION TERMINATED */
	if (K >= M)
	    break;
	/*
	 * C     ROW AND COLUMN INTERCHANGE AND PIVOT ROW REDUCTION IN MATRIX A.
	 * C     ELEMENTS OF PIVOT COLUMN ARE SAVED IN AUXILIARY VECTOR AUX.
	 */
	LR = LST + (LT * (K + J - 1)) / 2;
	LL = LR;
	L = LST;
	for (II = K; II <= LEND; II++) {
	    L += II;
	    LL += 1;
	    if (L == LR) {
		A[LL - 1] = A[LST - 1];
		tb = A[L - 1];
		goto lab13;
	    }
	    if (L > LR)
		LL = L + LT;

	    tb = A[LL - 1];
	    A[LL - 1] = A[L - 1];
	  lab13:
	    AUX[II - 1] = tb;
	    A[L - 1] = pivi * tb;
	}
	/* SAVE COLUMN INTERCHANGE INFORMATION */
	A[LST - 1] = LT;
	/* ELEMENT REDUCTION AND SEARCH FOR NEXT PIVOT */
	piv = 0.0;
	LLST = LST;
	LT = 0;
	for (II = K; II <= LEND; II++) {
	    pivi = -AUX[II - 1];
	    LL = LLST;
	    LT += 1;
	    for (LLD = II; LLD <= LEND; LLD++) {
		LL += LLD;
		L = LL + LT;
		A[L - 1] += pivi * A[LL - 1];
	    }
	    LLST += II;
	    LR = LLST + LT;
	    tb = fabs(A[LR - 1]);
	    if (tb > piv) {
		piv = tb;
		I = LR;
		J = II + 1;
	    }
	    LR = K;
	    LL = LR + LT;
	    R[LL - 1] += pivi * R[LR - 1];
	}
    }
    /* END OF ELIMINATION LOOP */

    /* BACK SUBSTITUTION AND BACK INTERCHANGE */

    if (LEND <= 0) {
	if (LEND < 0)
	    goto fatal;
	goto done;
    }
    II = M;
    for (I = 2; I <= M; I++) {
	LST -= II;
	II -= 1;
	L = A[LST - 1] + 0.5;
	J = II;
	tb = R[J - 1];
	LL = J;
	K = LST;
	for (LT = II; LT <= LEND; LT++) {
	    LL += 1;
	    K += LT;
	    tb -= A[K - 1] * R[LL - 1];
	}
	K = J + L;
	R[J - 1] = R[K - 1];
	R[K - 1] = tb;
    }
  done:
    return (IER);
}
