/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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
static char sccsid[] = "@(#)creat.c	8.1 (Berkeley) 6/2/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/compat-43/creat.c,v 1.8 2007/01/09 00:27:49 imp Exp $");


#include "namespace.h"
#include <fcntl.h>
#include "un-namespace.h"

#ifdef VARIANT_CANCELABLE
int __open(const char *path, int flags, mode_t mode);
#else /* !VARIANT_CANCELABLE */
int __open_nocancel(const char *path, int flags, mode_t mode);
#endif /* VARIANT_CANCELABLE */


int
__creat(const char *path, mode_t mode)
{
#ifdef VARIANT_CANCELABLE
	return(__open(path, O_WRONLY|O_CREAT|O_TRUNC, mode));
#else /* !VARIANT_CANCELABLE */
	return(__open_nocancel(path, O_WRONLY|O_CREAT|O_TRUNC, mode));
#endif /* VARIANT_CANCELABLE */
}
__weak_reference(__creat, creat);
__weak_reference(__creat, _creat);
