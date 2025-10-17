/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
static char sccsid[] = "@(#)pause.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */

#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/pause.c,v 1.8 2009/12/05 19:31:38 ed Exp $");

#include <signal.h>
#include <unistd.h>

/*
 * Backwards compatible pause.
 */
int
__pause(void)
{
	sigset_t set;

	sigprocmask(0, NULL, &set);
	return sigsuspend(&set);
}
__weak_reference(__pause, pause);
__weak_reference(__pause, _pause);
