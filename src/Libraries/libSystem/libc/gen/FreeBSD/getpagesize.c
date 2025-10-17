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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)getpagesize.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/getpagesize.c,v 1.6 2007/01/09 00:27:54 imp Exp $");

#include <sys/param.h>
#include <sys/sysctl.h>

#include <unistd.h>

/*
 * This is unlikely to change over the running time of any
 * program, so we cache the result to save some syscalls.
 *
 * NB: This function may be called from malloc(3) at initialization
 * NB: so must not result in a malloc(3) related call!
 */

int
getpagesize()
{
	int mib[2]; 
	static int value;
	size_t size;

	if (!value) {
		mib[0] = CTL_HW;
		mib[1] = HW_PAGESIZE;
		size = sizeof value;
		if (sysctl(mib, 2, &value, &size, NULL, 0) == -1)
			return (-1);
	}
	return (value);
}
