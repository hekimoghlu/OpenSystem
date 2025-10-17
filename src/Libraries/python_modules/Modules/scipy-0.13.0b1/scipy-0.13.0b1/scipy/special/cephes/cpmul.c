/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
/*                                                     cpmul   */

typedef struct {
    double r;
    double i;
} cmplx;

void cpmul(cmplx *, int, cmplx *, int, cmplx *, int *);

void cpmul(a, da, b, db, c, dc)
cmplx *a, *b, *c;
int da, db;
int *dc;
{
    int i, j, k;
    cmplx y;
    register cmplx *pa, *pb, *pc;

    if (da > db) {		/* Know which polynomial has higher degree */
	i = da;			/* Swapping is OK because args are on the stack */
	da = db;
	db = i;
	pa = a;
	a = b;
	b = pa;
    }

    k = da + db;
    *dc = k;			/* Output the degree of the product */
    pc = &c[db + 1];
    for (i = db + 1; i <= k; i++) {	/* Clear high order terms of output */
	pc->r = 0;
	pc->i = 0;
	pc++;
    }
    /* To permit replacement of input, work backward from highest degree */
    pb = &b[db];
    for (j = 0; j <= db; j++) {
	pa = &a[da];
	pc = &c[k - j];
	for (i = 0; i < da; i++) {
	    y.r = pa->r * pb->r - pa->i * pb->i;	/* cmpx multiply */
	    y.i = pa->r * pb->i + pa->i * pb->r;
	    pc->r += y.r;	/* accumulate partial product */
	    pc->i += y.i;
	    pa--;
	    pc--;
	}
	y.r = pa->r * pb->r - pa->i * pb->i;	/* replace last term,   */
	y.i = pa->r * pb->i + pa->i * pb->r;	/* ...do not accumulate */
	pc->r = y.r;
	pc->i = y.i;
	pb--;
    }
}
