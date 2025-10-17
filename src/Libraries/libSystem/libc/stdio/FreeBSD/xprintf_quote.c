/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
#include <ctype.h>
#include <wchar.h>
#include <vis.h>
#include <assert.h>
#include <sys/time.h>
#include "printf.h"
#include "xprintf_private.h"

__private_extern__ int
__printf_arginfo_quote(const struct printf_info *pi __unused, size_t n, int *argt)
{

	assert(n >= 1);
	argt[0] = PA_POINTER;
	return (1);
}

__private_extern__ int
__printf_render_quote(struct __printf_io *io, const struct printf_info *pi __unused, const void *const *arg)
{
	const char *str, *p, *t, *o;
	char r[5];
	int i, ret;

	str = *((const char *const *)arg[0]);
	if (str == NULL)
		return (__printf_out(io, pi, "\"(null)\"", 8));
	if (*str == '\0')
		return (__printf_out(io, pi, "\"\"", 2));

	for (i = 0, p = str; *p; p++)
		if (isspace(*p) || *p == '\\' || *p == '"')
			i++;
	if (!i) 
		return (__printf_out(io, pi, str, strlen(str)));
	
	ret = __printf_out(io, pi, "\"", 1);
	for (t = p = str; *p; p++) {
		o = NULL;
		if (*p == '\\')
			o = "\\\\";
		else if (*p == '\n')
			o = "\\n";
		else if (*p == '\r')
			o = "\\r";
		else if (*p == '\t')
			o = "\\t";
		else if (*p == ' ')
			o = " ";
		else if (*p == '"')
			o = "\\\"";
		else if (isspace(*p)) {
			sprintf(r, "\\%03o", *p);
			o = r;
		} else
			continue;
		if (p != t)
			ret += __printf_out(io, pi, t, p - t);
		ret += __printf_out(io, pi, o, strlen(o));
		t = p + 1;
	}
	if (p != t)
		ret += __printf_out(io, pi, t, p - t);
	ret += __printf_out(io, pi, "\"", 1);
	__printf_flush(io);
	return(ret);
}
