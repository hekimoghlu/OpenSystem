/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wstrict-prototypes"

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)exit.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdlib/exit.c,v 1.9 2007/01/09 00:28:09 imp Exp $");

#include "namespace.h"
#include <stdlib.h>
#include <unistd.h>
#include "un-namespace.h"

#include "atexit.h"
#include "libc_private.h"
#include "stdio/FreeBSD/local.h" // for __cleanup

#include <TargetConditionals.h>

#if __APPLE__
int __cleanup = 0;
#else
void (* CLEANUP_PTRAUTH __cleanup)(void);
#endif // __APPLE__

extern void __exit(int) __attribute__((noreturn));
#if __has_feature(cxx_thread_local) || __has_feature(c_thread_local)
extern void _tlv_exit();
#endif // __has_feature(cxx_thread_local) || __has_feature(c_thread_local)

/*
 * Exit, flushing stdio buffers if necessary.
 */
void
exit(int status)
{
#if __has_feature(cxx_thread_local) || __has_feature(c_thread_local)
	_tlv_exit(); // C++11 requires thread_local objects to be destroyed before global objects
#endif // __has_feature(cxx_thread_local) || __has_feature(c_thread_local)
	__cxa_finalize(NULL);
	if (__cleanup)
#ifdef __APPLE__
		_cleanup();
#else
		(*__cleanup)();
#endif // __APPLE__
	__exit(status);
}
#pragma clang diagnostic pop
