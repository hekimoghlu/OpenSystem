/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
__printf_arginfo_errno(const struct printf_info *pi __unused, size_t n, int *argt)
{

	assert(n >= 1);
	argt[0] = PA_INT;
	return (1);
}

__private_extern__ int
__printf_render_errno(struct __printf_io *io, const struct printf_info *pi __unused, const void *const *arg)
{
	int ret, error;
	char buf[64];
	const char *p;

	ret = 0;
	error = *((const int *)arg[0]);
	if (error >= 0 && error < sys_nerr) {
		p = strerror(error);
		return (__printf_out(io, pi, p, strlen(p)));
	}
	sprintf(buf, "errno=%d/0x%x", error, error);
	ret += __printf_out(io, pi, buf, strlen(buf));
	__printf_flush(io);
	return(ret);
}
