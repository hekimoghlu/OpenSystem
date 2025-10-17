/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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

#include <TargetConditionals.h>

void (*__cleanup)(void);
extern void __exit(int) __attribute__((noreturn));
#if !TARGET_IPHONE_SIMULATOR && (__i386__ || __x86_64__)
extern void _tlv_exit();
#endif

/*
 * Exit, flushing stdio buffers if necessary.
 */
void
exit(int status)
{
#if !TARGET_IPHONE_SIMULATOR && (__i386__ || __x86_64__)
	_tlv_exit(); // C++11 requires thread_local objects to be destroyed before global objects
#endif
	__cxa_finalize(NULL);
	if (__cleanup)
		(*__cleanup)();
	__exit(status);
}
#pragma clang diagnostic pop
