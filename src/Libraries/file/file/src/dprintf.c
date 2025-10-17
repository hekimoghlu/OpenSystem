/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
#include "file.h"

#ifndef	lint
FILE_RCSID("@(#)$File: dprintf.c,v 1.2 2018/09/09 20:33:28 christos Exp $")
#endif	/* lint */

#include <assert.h>
#include <unistd.h>
#include <stdio.h>
#include <stdarg.h>

int
dprintf(int fd, const char *fmt, ...)
{
	va_list ap;
	/* Simpler than using vasprintf() here, since we never need more */
	char buf[1024];
	int len;

	va_start(ap, fmt);
	len = vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);

	if ((size_t)len >= sizeof(buf))
		return -1;

	if (write(fd, buf, (size_t)len) != len)
		return -1;

	return len;
}
