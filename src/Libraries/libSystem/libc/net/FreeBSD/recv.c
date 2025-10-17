/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
static char sccsid[] = "@(#)recv.c	8.2 (Berkeley) 2/21/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/net/recv.c,v 1.4 2007/01/09 00:28:02 imp Exp $");

#include "namespace.h"
#include <sys/types.h>
#include <sys/socket.h>

#include <stddef.h>
#include "un-namespace.h"

#ifdef VARIANT_CANCELABLE
ssize_t __recvfrom(int, void *, size_t, int, struct sockaddr * __restrict, socklen_t * __restrict);
#else /* !VARIANT_CANCELABLE */
ssize_t __recvfrom_nocancel(int, void *, size_t, int, struct sockaddr * __restrict, socklen_t * __restrict);
#endif /* VARIANT_CANCELABLE */

ssize_t
recv(s, buf, len, flags)
	int s, flags;
	size_t len;
	void *buf;
{
#ifdef VARIANT_CANCELABLE
	return (__recvfrom(s, buf, len, flags, NULL, 0));
#else /* !VARIANT_CANCELABLE */
	return (__recvfrom_nocancel(s, buf, len, flags, NULL, 0));
#endif /* VARIANT_CANCELABLE */
}
