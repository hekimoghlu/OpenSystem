/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
static char sccsid[] = "@(#)wait.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/wait.c,v 1.7 2007/01/09 00:27:56 imp Exp $");

#include "namespace.h"
#include <sys/types.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include "un-namespace.h"

#ifdef VARIANT_CANCELABLE
int __wait4(pid_t, int *, int , struct rusage *);
#else /* !VARIANT_CANCELABLE */
int __wait4_nocancel(pid_t, int *, int , struct rusage *);
#endif /* VARIANT_CANCELABLE */

pid_t
__wait(int *istat)
{
#ifdef VARIANT_CANCELABLE
	return (__wait4(WAIT_ANY, istat, 0, (struct rusage *)0));
#else /* !VARIANT_CANCELABLE */
	return (__wait4_nocancel(WAIT_ANY, istat, 0, (struct rusage *)0));
#endif /* VARIANT_CANCELABLE */
}

__weak_reference(__wait, wait);
__weak_reference(__wait, _wait);
