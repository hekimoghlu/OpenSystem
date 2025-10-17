/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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
 * Glenn Fowler
 * AT&T Bell Laboratories
 *
 * return strperm() expression for perm
 */

#include <ast.h>
#include <ls.h>

char*
fmtperm(register int perm)
{
	register char*	s;
	char*		buf;

	s = buf = fmtbuf(32);

	/*
	 * u
	 */

	*s++ = 'u';
	*s++ = '=';
	if (perm & S_ISVTX)
		*s++ = 't';
	if (perm & S_ISUID)
		*s++ = 's';
	if (perm & S_IRUSR)
		*s++ = 'r';
	if (perm & S_IWUSR)
		*s++ = 'w';
	if (perm & S_IXUSR)
		*s++ = 'x';
	if ((perm & (S_ISGID|S_IXGRP)) == S_ISGID)
		*s++ = 'l';

	/*
	 * g
	 */

	*s++ = ',';
	*s++ = 'g';
	*s++ = '=';
	if ((perm & (S_ISGID|S_IXGRP)) == (S_ISGID|S_IXGRP))
		*s++ = 's';
	if (perm & S_IRGRP)
		*s++ = 'r';
	if (perm & S_IWGRP)
		*s++ = 'w';
	if (perm & S_IXGRP)
		*s++ = 'x';

	/*
	 * o
	 */

	*s++ = ',';
	*s++ = 'o';
	*s++ = '=';
	if (perm & S_IROTH)
		*s++ = 'r';
	if (perm & S_IWOTH)
		*s++ = 'w';
	if (perm & S_IXOTH)
		*s++ = 'x';
	*s = 0;
	return buf;
}
