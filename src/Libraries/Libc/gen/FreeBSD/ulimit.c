/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <ulimit.h>

long
ulimit(int cmd, ...)
{
	struct rlimit limit;
	va_list ap;
	long arg;

	if (cmd == UL_GETFSIZE) {
		if (getrlimit(RLIMIT_FSIZE, &limit) == -1)
			return (-1);
		limit.rlim_cur /= 512;
		if (limit.rlim_cur > LONG_MAX)
			return (LONG_MAX);
		return ((long)limit.rlim_cur);
	} else if (cmd == UL_SETFSIZE) {
		va_start(ap, cmd);
		arg = va_arg(ap, long);
		va_end(ap);
		limit.rlim_max = limit.rlim_cur = (rlim_t)arg * 512;

		/* The setrlimit() function sets errno to EPERM if needed. */
		if (setrlimit(RLIMIT_FSIZE, &limit) == -1)
			return (-1);
		if (arg * 512 > LONG_MAX)
			return (LONG_MAX);
		return (arg);
	} else {
		errno = EINVAL;
		return (-1);
	}
}

