/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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
#pragma prototyped
/*
 *  strsort - sort an array pointers using fn
 *
 *	fn follows strcmp(3) conventions
 *
 *   David Korn
 *   AT&T Bell Laboratories
 *
 *  derived from Bourne Shell
 */

#include <ast.h>

void
strsort(char** argv, int n, int(*fn)(const char*, const char*))
{
	register int 	i;
	register int 	j;
	register int 	m;
	register char**	ap;
	char*		s;
	int 		k;

	for (j = 1; j <= n; j *= 2);
	for (m = 2 * j - 1; m /= 2;)
		for (j = 0, k = n - m; j < k; j++)
			for (i = j; i >= 0; i -= m)
			{
				ap = &argv[i];
				if ((*fn)(ap[m], ap[0]) >= 0) break;
				s = ap[m];
				ap[m] = ap[0];
				ap[0] = s;
			}
}
