/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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
 * return ls -l style file mode string given file mode bits
 * if external!=0 then mode is modex canonical
 */

#include "modelib.h"

char*
fmtmode(register int mode, int external)
{
	register char*		s;
	register struct modeop*	p;
	char*			buf;

	if (!external)
		mode = modex(mode);
	s = buf = fmtbuf(MODELEN + 1);
	for (p = modetab; p < &modetab[MODELEN]; p++)
		*s++ = p->name[((mode & p->mask1) >> p->shift1) | ((mode & p->mask2) >> p->shift2)];
	*s = 0;
	return buf;
}
