/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
#include "slu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    mark_relax() - record the rows used by the relaxed supernodes.
 * </pre>
 */
int mark_relax(
	int n,		    /* order of the matrix A */
	int *relax_end,     /* last column in a relaxed supernode.
			     * if j-th column starts a relaxed supernode,
			     * relax_end[j] represents the last column of
			     * this supernode. */
	int *relax_fsupc,   /* first column in a relaxed supernode.
			     * relax_fsupc[j] represents the first column of
			     * j-th supernode. */
	int *xa_begin,	    /* Astore->colbeg */
	int *xa_end,	    /* Astore->colend */
	int *asub,	    /* row index of A */
	int *marker	    /* marker[j] is the maximum column index if j-th
			     * row belongs to a relaxed supernode. */ )
{
    register int jcol, kcol;
    register int i, j, k;

    for (i = 0; i < n && relax_fsupc[i] != EMPTY; i++)
    {
	jcol = relax_fsupc[i];	/* first column */
	kcol = relax_end[jcol]; /* last column */
	for (j = jcol; j <= kcol; j++)
	    for (k = xa_begin[j]; k < xa_end[j]; k++)
		marker[asub[k]] = jcol;
    }
    return i;
}
