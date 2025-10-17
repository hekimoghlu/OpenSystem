/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
#if defined(VARIANT_CANCELABLE) && __DARWIN_NON_CANCELABLE != 0
#error cancellable call vs. __DARWIN_NON_CANCELABLE mismatch
#endif

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)usleep.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/usleep.c,v 1.31 2009/12/05 19:31:38 ed Exp $");

#include "namespace.h"
#include <time.h>
#include <unistd.h>
#include "un-namespace.h"

int
usleep(useconds_t useconds)
{
	struct timespec time_to_sleep;

	time_to_sleep.tv_nsec = (useconds % 1000000) * 1000;
	time_to_sleep.tv_sec = useconds / 1000000;
	return (_nanosleep(&time_to_sleep, NULL));
}
