/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
#include <sys/cdefs.h>
#include <sys/types.h>
#include <stdarg.h>
#include <sys/fcntl.h>
#include <sys/guarded.h>

int __guarded_open_dprotected_np(const char *path,
    const guardid_t *guard, u_int guardflags, int flags, int dpclass, int dpflags, int mode);

int
guarded_open_dprotected_np(const char *path,
    const guardid_t *guard, u_int guardflags, int flags, int dpclass, int dpflags, ...)
{
	int mode = 0;

	if (flags & O_CREAT) {
		va_list ap;
		va_start(ap, dpflags);
		mode = va_arg(ap, int);
		va_end(ap);
	}
	return __guarded_open_dprotected_np(path, guard, guardflags, flags, dpclass, dpflags, mode);
}
