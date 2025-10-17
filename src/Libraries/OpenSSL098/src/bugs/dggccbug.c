/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
/* dggccbug.c */
/* bug found by Eric Young (eay@cryptsoft.com) - May 1995 */

#include <stdio.h>

/* There is a bug in
 * gcc version 2.5.8 (88open OCS/BCS, DG-2.5.8.3, Oct 14 1994)
 * as shipped with DGUX 5.4R3.10 that can be bypassed by defining
 * DG_GCC_BUG in my code.
 * The bug manifests itself by the vaule of a pointer that is
 * used only by reference, not having it's value change when it is used
 * to check for exiting the loop.  Probably caused by there being 2
 * copies of the valiable, one in a register and one being an address
 * that is passed. */

/* compare the out put from
 * gcc dggccbug.c; ./a.out
 * and
 * gcc -O dggccbug.c; ./a.out
 * compile with -DFIXBUG to remove the bug when optimising.
 */

void inc(a)
int *a;
	{
	(*a)++;
	}

main()
	{
	int p=0;
#ifdef FIXBUG
	int dummy;
#endif

	while (p<3)
		{
		fprintf(stderr,"%08X\n",p);
		inc(&p);
#ifdef FIXBUG
		dummy+=p;
#endif
		}
	}
