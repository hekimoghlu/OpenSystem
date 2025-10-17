/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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

double dqselect(int n, double A[], int k)
{
    register int i, j, p;
    register double val;

    k = SUPERLU_MAX(k, 0);
    k = SUPERLU_MIN(k, n - 1);
    while (n > 1)
    {
	i = 0; j = n-1;
	p = j; val = A[p];
	while (i < j)
	{
	    for (; A[i] >= val && i < p; i++);
	    if (A[i] < val) { A[p] = A[i]; p = i; }
	    for (; A[j] <= val && j > p; j--);
	    if (A[j] > val) { A[p] = A[j]; p = j; }
	}
	A[p] = val;
	if (p == k) return val;
	else if (p > k) n = p;
	else
	{
	    p++;
	    n -= p; A += p; k -= p;
	}
    }

    return A[0];
}

float sqselect(int n, float A[], int k)
{
    register int i, j, p;
    register float val;

    k = SUPERLU_MAX(k, 0);
    k = SUPERLU_MIN(k, n - 1);
    while (n > 1)
    {
	i = 0; j = n-1;
	p = j; val = A[p];
	while (i < j)
	{
	    for (; A[i] >= val && i < p; i++);
	    if (A[i] < val) { A[p] = A[i]; p = i; }
	    for (; A[j] <= val && j > p; j--);
	    if (A[j] > val) { A[p] = A[j]; p = j; }
	}
	A[p] = val;
	if (p == k) return val;
	else if (p > k) n = p;
	else
	{
	    p++;
	    n -= p; A += p; k -= p;
	}
    }

    return A[0];
}
