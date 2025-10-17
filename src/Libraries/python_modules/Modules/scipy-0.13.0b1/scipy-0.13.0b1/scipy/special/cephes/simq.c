/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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
/*                                                     simq    2 */

#include <stdio.h>
int simq(double[], double[], double[], int, int, int[]);

#define fabs(x) ((x) < 0 ? -(x) : (x))

int simq(A, B, X, n, flag, IPS)
double A[], B[], X[];
int n, flag;
int IPS[];
{
    int i, j, ij, ip, ipj, ipk, ipn;
    int idxpiv, iback;
    int k, kp, kp1, kpk, kpn;
    int nip, nkp, nm1;
    double em, q, rownrm, big, size, pivot, sum;

    nm1 = n - 1;
    if (flag < 0)
	goto solve;

    /*     Initialize IPS and X    */

    ij = 0;
    for (i = 0; i < n; i++) {
	IPS[i] = i;
	rownrm = 0.0;
	for (j = 0; j < n; j++) {
	    q = fabs(A[ij]);
	    if (rownrm < q)
		rownrm = q;
	    ++ij;
	}
	if (rownrm == 0.0) {
	    puts("SIMQ ROWNRM=0");
	    return (1);
	}
	X[i] = 1.0 / rownrm;
    }

    /*                                                     simq    3 */
    /*     Gaussian elimination with partial pivoting      */

    for (k = 0; k < nm1; k++) {
	big = 0.0;
	idxpiv = 0;
	for (i = k; i < n; i++) {
	    ip = IPS[i];
	    ipk = n * ip + k;
	    size = fabs(A[ipk]) * X[ip];
	    if (size > big) {
		big = size;
		idxpiv = i;
	    }
	}

	if (big == 0.0) {
	    puts("SIMQ BIG=0");
	    return (2);
	}
	if (idxpiv != k) {
	    j = IPS[k];
	    IPS[k] = IPS[idxpiv];
	    IPS[idxpiv] = j;
	}
	kp = IPS[k];
	kpk = n * kp + k;
	pivot = A[kpk];
	kp1 = k + 1;
	for (i = kp1; i < n; i++) {
	    ip = IPS[i];
	    ipk = n * ip + k;
	    em = -A[ipk] / pivot;
	    A[ipk] = -em;
	    nip = n * ip;
	    nkp = n * kp;
	    for (j = kp1; j < n; j++) {
		ipj = nip + j;
		A[ipj] = A[ipj] + em * A[nkp + j];
	    }
	}
    }
    kpn = n * IPS[n - 1] + n - 1;	/* last element of IPS[n] th row */
    if (A[kpn] == 0.0) {
	puts("SIMQ A[kpn]=0");
	return (3);
    }

    /*                                                     simq 4 */
    /*     back substitution       */

  solve:
    ip = IPS[0];
    X[0] = B[ip];
    for (i = 1; i < n; i++) {
	ip = IPS[i];
	ipj = n * ip;
	sum = 0.0;
	for (j = 0; j < i; j++) {
	    sum += A[ipj] * X[j];
	    ++ipj;
	}
	X[i] = B[ip] - sum;
    }

    ipn = n * IPS[n - 1] + n - 1;
    X[n - 1] = X[n - 1] / A[ipn];

    for (iback = 1; iback < n; iback++) {
	/* i goes (n-1),...,1  */
	i = nm1 - iback;
	ip = IPS[i];
	nip = n * ip;
	sum = 0.0;
	for (j = i + 1; j < n; j++)
	    sum += A[nip + j] * X[j];
	X[i] = (X[i] - sum) / A[nip + i];
    }
    return (0);
}
