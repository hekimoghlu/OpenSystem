/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#include "slu_sdefs.h"

int sfill_diag(int n, NCformat *Astore)
/* fill explicit zeros on the diagonal entries, so that the matrix is not
   structurally singular. */
{
    float *nzval = (float *)Astore->nzval;
    int *rowind = Astore->rowind;
    int *colptr = Astore->colptr;
    int nnz = colptr[n];
    int fill = 0;
    float *nzval_new;
    float zero = 0.0;
    int *rowind_new;
    int i, j, diag;

    for (i = 0; i < n; i++)
    {
	diag = -1;
	for (j = colptr[i]; j < colptr[i + 1]; j++)
	    if (rowind[j] == i) diag = j;
	if (diag < 0) fill++;
    }
    if (fill)
    {
	nzval_new = floatMalloc(nnz + fill);
	rowind_new = intMalloc(nnz + fill);
	fill = 0;
	for (i = 0; i < n; i++)
	{
	    diag = -1;
	    for (j = colptr[i] - fill; j < colptr[i + 1]; j++)
	    {
		if ((rowind_new[j + fill] = rowind[j]) == i) diag = j;
		nzval_new[j + fill] = nzval[j];
	    }
	    if (diag < 0)
	    {
		rowind_new[colptr[i + 1] + fill] = i;
		nzval_new[colptr[i + 1] + fill] = zero;
		fill++;
	    }
	    colptr[i + 1] += fill;
	}
	Astore->nzval = nzval_new;
	Astore->rowind = rowind_new;
	SUPERLU_FREE(nzval);
	SUPERLU_FREE(rowind);
    }
    Astore->nnz += fill;
    return fill;
}

int sdominate(int n, NCformat *Astore)
/* make the matrix diagonally dominant */
{
    float *nzval = (float *)Astore->nzval;
    int *rowind = Astore->rowind;
    int *colptr = Astore->colptr;
    int nnz = colptr[n];
    int fill = 0;
    float *nzval_new;
    int *rowind_new;
    int i, j, diag;
    double s;

    for (i = 0; i < n; i++)
    {
	diag = -1;
	for (j = colptr[i]; j < colptr[i + 1]; j++)
	    if (rowind[j] == i) diag = j;
	if (diag < 0) fill++;
    }
    if (fill)
    {
	nzval_new = floatMalloc(nnz + fill);
	rowind_new = intMalloc(nnz+ fill);
	fill = 0;
	for (i = 0; i < n; i++)
	{
	    s = 1e-6;
	    diag = -1;
	    for (j = colptr[i] - fill; j < colptr[i + 1]; j++)
	    {
		if ((rowind_new[j + fill] = rowind[j]) == i) diag = j;
		s += fabs(nzval_new[j + fill] = nzval[j]);
	    }
	    if (diag >= 0) {
		nzval_new[diag+fill] = s * 3.0;
	    } else {
		rowind_new[colptr[i + 1] + fill] = i;
		nzval_new[colptr[i + 1] + fill] = s * 3.0;
		fill++;
	    }
	    colptr[i + 1] += fill;
	}
	Astore->nzval = nzval_new;
	Astore->rowind = rowind_new;
	SUPERLU_FREE(nzval);
	SUPERLU_FREE(rowind);
    }
    else
    {
	for (i = 0; i < n; i++)
	{
	    s = 1e-6;
	    diag = -1;
	    for (j = colptr[i]; j < colptr[i + 1]; j++)
	    {
		if (rowind[j] == i) diag = j;
		s += fabs(nzval[j]);
	    }
	    nzval[diag] = s * 3.0;
	}
    }
    Astore->nnz += fill;
    return fill;
}
