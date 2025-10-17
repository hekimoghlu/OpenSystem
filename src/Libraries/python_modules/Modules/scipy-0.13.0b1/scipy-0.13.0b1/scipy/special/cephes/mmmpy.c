/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 15, 2024.
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

void mmmpy(r, c, A, B, Y)
int r, c;
double *A, *B, *Y;
{
    register double s;
    double *pA, *pB, *pY, *pt;
    int i, j, k;

    pY = Y;
    pB = B;
    for (i = 0; i < r; i++) {
	pA = A;
	for (j = 0; j < r; j++) {
	    pt = pB;
	    s = 0.0;
	    for (k = 0; k < c; k++) {
		s += *pA++ * *pt;
		pt += r;	/* increment to next row underneath */
	    }
	    *pY++ = s;
	}
	pB += 1;
    }
}
