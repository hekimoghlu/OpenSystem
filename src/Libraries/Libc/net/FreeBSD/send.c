/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
static char sccsid[] = "@(#)send.c	8.2 (Berkeley) 2/21/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/net/send.c,v 1.4 2007/01/09 00:28:02 imp Exp $");

#include "namespace.h"
#include <sys/types.h>
#include <sys/socket.h>

#include <stddef.h>
#include "un-namespace.h"

#ifdef VARIANT_CANCELABLE
ssize_t __sendto(int, const void *, size_t, int, const struct sockaddr *, socklen_t);
#else /* !VARIANT_CANCELABLE */
ssize_t __sendto_nocancel(int, const void *, size_t, int, const struct sockaddr *, socklen_t);
#endif /* VARIANT_CANCELABLE */

ssize_t
send(int s, const void *msg, size_t len, int flags)
{
#ifdef VARIANT_CANCELABLE
	return (__sendto(s, msg, len, flags, NULL, 0));
#else /* !VARIANT_CANCELABLE */
	return (__sendto_nocancel(s, msg, len, flags, NULL, 0));
#endif /* VARIANT_CANCELABLE */
}
