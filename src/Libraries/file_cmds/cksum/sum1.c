/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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
#ifndef lint
#if 0
static char sccsid[] = "@(#)sum1.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* not lint */

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>

#include <unistd.h>
#include <stdint.h>

#include "extern.h"

int
csum1(int fd, uint32_t *cval, off_t *clen)
{
	int nr;
	u_int lcrc;
	off_t total;
	u_char *p;
	u_char buf[8192];

	/*
	 * 16-bit checksum, rotating right before each addition;
	 * overflow is discarded.
	 */
	lcrc = total = 0;
	while ((nr = read(fd, buf, sizeof(buf))) > 0)
		for (total += nr, p = buf; nr--; ++p) {
			if (lcrc & 1)
				lcrc |= 0x10000;
			lcrc = ((lcrc >> 1) + *p) & 0xffff;
		}
	if (nr < 0)
		return (1);

	*cval = lcrc;
	*clen = total;
	return (0);
}
