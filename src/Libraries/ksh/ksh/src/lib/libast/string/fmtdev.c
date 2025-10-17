/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
 * st_dev formatter
 */

#include <ast.h>
#include <ctype.h>
#include <ls.h>

char*
fmtdev(struct stat* st)
{
	char*		buf;
	unsigned long	mm;
	unsigned int	ma;
	unsigned int	mi;
	int		z;

	mm = (S_ISBLK(st->st_mode) || S_ISCHR(st->st_mode)) ? idevice(st) : st->st_dev;
	ma = major(mm);
	mi = minor(mm);
	buf = fmtbuf(z = 17);
	if (ma == '#' && isalnum(mi))
	{
		/*
		 * Plan? Nein!
		 */

		buf[0] = ma;
		buf[1] = mi;
		buf[2] = 0;
	}
	else
		sfsprintf(buf, z, "%03d,%03d", ma, mi);
	return buf;
}
