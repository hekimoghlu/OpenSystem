/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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
void mtransp(int, double *, double *);

void mtransp(n, A, T)
int n;
double *A, *T;
{
    int i, j, np1;
    double *pAc, *pAr, *pTc, *pTr, *pA0, *pT0;
    double x;

    np1 = n + 1;
    pA0 = A;
    pT0 = T;
    for (i = 0; i < n - 1; i++) {	/* row index */
	pAc = pA0;		/* next diagonal element of input */
	pAr = pAc + n;		/* next row down underneath the diagonal element */
	pTc = pT0;		/* next diagonal element of the output */
	pTr = pTc + n;		/* next row underneath */
	*pTc++ = *pAc++;	/* copy the diagonal element */
	for (j = i + 1; j < n; j++) {	/* column index */
	    x = *pAr;
	    *pTr = *pAc++;
	    *pTc++ = x;
	    pAr += n;
	    pTr += n;
	}
	pA0 += np1;		/* &A[n*i+i] for next i */
	pT0 += np1;		/* &T[n*i+i] for next i */
    }
    *pT0 = *pA0;		/* copy the diagonal element */
}
