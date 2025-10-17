/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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
#include <namespace.h>
#include <stdio.h>
#include <wchar.h>
#include <stdint.h>
#include <assert.h>
#include <sys/time.h>
#include "printf.h"
#include "xprintf_private.h"

__private_extern__ int
__printf_arginfo_hexdump(const struct printf_info *pi, size_t n, int *argt)
{

	assert(n >= 2);
	argt[0] = PA_POINTER;
	argt[1] = PA_INT;
	return (2);
}

__private_extern__ int
__printf_render_hexdump(struct __printf_io *io, const struct printf_info *pi, const void *const *arg)
{
	unsigned char *p;
	unsigned u, l, j, a;
	char buf[100], *q;
	int ret;

	if (pi->width > 0 && pi->width < 16)
		l = pi->width;
	else
		l = 16;
	p = *((unsigned char **)arg[0]);
	u = *((unsigned *)arg[1]);

	ret = 0;
	a = 0;
	while (u > 0) {
		q = buf;
		if (pi->showsign)
			q += sprintf(q, " %04x", a);
		for (j = 0; j < l && j < u; j++)
			q += sprintf(q, " %02x", p[j]);
		if (pi->alt) {
			for (; j < l; j++)
				q += sprintf(q, "   ");
			q += sprintf(q, "  |");
			for (j = 0; j < l && j < u; j++) {
				if (p[j] < ' ' || p[j] > '~')
					*q++ = '.';
				else
					*q++ = p[j];
			}
			for (; j < l; j++)
				*q++ = ' ';
			*q++ = '|';
		}
		if (l < u)
			j = l;
		else
			j = u;
		p += j;
		u -= j;
		a += j;
		if (u > 0)
			*q++ = '\n';
		ret += __printf_puts(io, buf + 1, q - (buf + 1));
		__printf_flush(io);
	}
	return (ret);
}
