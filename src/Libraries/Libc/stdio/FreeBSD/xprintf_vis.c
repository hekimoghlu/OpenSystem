/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
#include <string.h>
#include <stdlib.h>
#include <wchar.h>
#include <vis.h>
#include <assert.h>
#include <sys/time.h>
#include "printf.h"
#include "xprintf_private.h"

__private_extern__ int
__printf_arginfo_vis(const struct printf_info *pi, size_t n, int *argt)
{

	assert(n >= 1);
	argt[0] = PA_POINTER;
	return (1);
}

__private_extern__ int
__printf_render_vis(struct __printf_io *io, const struct printf_info *pi, const void *const *arg)
{
	char *p, *buf;
	unsigned l;
	int ret;

	ret = 0;
	p = *((char **)arg[0]);
	if (p == NULL)
		return (__printf_out(io, pi, "(null)", 6));
	if (pi->prec >= 0)
		l = pi->prec;
	else
		l = strlen(p);
	buf = MALLOC(l * 4 + 1);
	if (buf == NULL)
		return (-1);
	if (pi->showsign)
		ret = strvisx(buf, p, l, VIS_WHITE | VIS_HTTPSTYLE);
	else if (pi->pad == '0')
		ret = strvisx(buf, p, l, VIS_WHITE | VIS_OCTAL);
	else if (pi->alt)
		ret = strvisx(buf, p, l, VIS_WHITE);
	else
		ret = strvisx(buf, p, l, VIS_WHITE | VIS_CSTYLE | VIS_OCTAL);
	ret += __printf_out(io, pi, buf, ret);
	__printf_flush(io);
	free(buf);
	return(ret);
}
