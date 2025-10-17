/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
 * mode_t representation support
 */

#include "modelib.h"

/*
 * convert internal mode to external
 */

#undef	modex

int
modex(register int i)
{
#if _S_IDPERM && _S_IDTYPE
	return(i);
#else
	register int	x;
	register int	c;

	x = 0;
#if _S_IDPERM
	x |= (i & 07777);
#else
	for (c = 0; c < PERMLEN; c++)
		if (i & permmap[c++])
			x |= permmap[c];
#endif
#if _S_IDTYPE
	x |= (i & X_IFMT);
#else
	if (S_ISREG(i)) x |= X_IFREG;
	else if (S_ISDIR(i)) x |= X_IFDIR;
#ifdef S_ISLNK
	else if (S_ISLNK(i)) x |= X_IFLNK;
#endif
	else if (S_ISBLK(i)) x |= X_IFBLK;
	else if (S_ISCHR(i)) x |= X_IFCHR;
#ifdef S_ISCTG
	else if (S_ISCTG(i)) x |= X_IFCTG;
#endif
	else if (S_ISFIFO(i)) x |= X_IFIFO;
#ifdef S_ISSOCK
	else if (S_ISSOCK(i)) x |= X_IFSOCK;
#endif
#endif
	return(x);
#endif
}
