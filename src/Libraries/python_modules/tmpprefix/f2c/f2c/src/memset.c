/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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
/* This is for the benefit of people whose systems don't provide
 * memset, memcpy, and memcmp.  If yours is such a system, adjust
 * the makefile by adding memset.o to the "OBJECTS =" assignment.
 * WARNING: the memcpy below is adequate for f2c, but is not a
 * general memcpy routine (which must correctly handle overlapping
 * fields).
 */

 int
#ifdef KR_headers
memcmp(s1, s2, n) char *s1, *s2; int n;
#else
memcmp(char *s1, char *s2, int n)
#endif
{
	char *se;

	for(se = s1 + n; s1 < se; s1++, s2++)
		if (*s1 != *s2)
			return *s1 - *s2;
	return 0;
	}

 char *
#ifdef KR_headers
memcpy(s1, s2, n) char *s1, *s2; int n;
#else
memcpy(char *s1, char *s2, int n)
#endif
{
	char *s0 = s1, *se = s1 + n;

	while(s1 < se)
		*s1++ = *s2++;
	return s0;
	}

 void
#ifdef KR_headers
memset(s, c, n) char *s; int c, n;
#else
memset(char *s, int c, int n)
#endif
{
	char *se = s + n;

	while(s < se)
		*s++ = c;
	}
