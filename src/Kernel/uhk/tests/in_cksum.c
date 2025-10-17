/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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
/*
 * Copyright (c) 1988, 1992, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 *	@(#)in_cksum.c	8.1 (Berkeley) 6/10/93
 */

#include <sys/param.h>
#include "in_cksum.h"

typedef union {
	char        c[2];
	u_short     s;
} short_union_t;

typedef union {
	u_short     s[2];
	long        l;
} long_union_t;

static __inline__ void
reduce(int * sum)
{
	long_union_t l_util;

	l_util.l = *sum;
	*sum = l_util.s[0] + l_util.s[1];
	if (*sum > 65535) {
		*sum -= 65535;
	}
	return;
}


#include <stdio.h>

unsigned short
in_cksum(void * pkt, int len)
{
	u_short * w;
	int sum = 0;

	w = (u_short *)pkt;
	while ((len -= 32) >= 0) {
		sum += w[0]; sum += w[1];
		sum += w[2]; sum += w[3];
		sum += w[4]; sum += w[5];
		sum += w[6]; sum += w[7];
		sum += w[8]; sum += w[9];
		sum += w[10]; sum += w[11];
		sum += w[12]; sum += w[13];
		sum += w[14]; sum += w[15];
		w += 16;
	}
	len += 32;
	while ((len -= 8) >= 0) {
		sum += w[0]; sum += w[1];
		sum += w[2]; sum += w[3];
		w += 4;
	}
	len += 8;
	if (len) {
		reduce(&sum);
		while ((len -= 2) >= 0) {
			sum += *w++;
		}
	}
	if (len == -1) { /* odd-length packet */
		short_union_t s_util;

		s_util.s = 0;
		s_util.c[0] = *((char *)w);
		s_util.c[1] = 0;
		sum += s_util.s;
	}
	reduce(&sum);
	return ~sum & 0xffff;
}
