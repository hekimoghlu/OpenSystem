/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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
#include "expect_cf.h"
#include "tcl.h"

/* like memcpy but can handle overlap */
#ifndef HAVE_MEMMOVE
char *
memmove(dest,src,n)
VOID *dest;
CONST VOID *src;
int n;
{
	char *d;
	CONST char *s;

	d = dest;
	s = src;
	if (s<d && (d < s+n)) {
		for (d+=n, s+=n; 0<n; --n)
			*--d = *--s;
	} else for (;0<n;--n) *d++ = *s++;
	return dest;
}
#endif /* HAVE_MEMMOVE */
