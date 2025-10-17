/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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
static char sccsid[] = "@(#)signal.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/signal.c,v 1.4 2007/01/09 00:27:55 imp Exp $");

/*
 * Almost backwards compatible signal.
 */
#include "namespace.h"
#include <signal.h>
#include "un-namespace.h"
#include "libc_private.h"

sigset_t _sigintr;		/* shared with siginterrupt */

extern int _sigaction_nobind (int sig, const struct sigaction *nsv, struct sigaction *osv);

static sig_t
signal__(s, a, bind)
	int s;
	sig_t a;
	int bind;
{
	struct sigaction sa, osa;

	sa.sa_handler = a;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = 0;
	if (!sigismember(&_sigintr, s))
		sa.sa_flags |= SA_RESTART;
#if defined(__DYNAMIC__)
	if (bind) {
#endif /* __DYNAMIC__ */
	if (_sigaction(s, &sa, &osa) < 0)
		return (SIG_ERR);
#if defined(__DYNAMIC__)
	} else {
	    if (_sigaction_nobind(s, &sa, &osa) < 0)
		return (SIG_ERR);
	}
#endif /* __DYNAMIC__ */
	return (osa.sa_handler);
}

sig_t
signal(s, a)
        int s;
        sig_t a;
{
    return signal__(s, a, 1);
}

#if defined(__DYNAMIC__)
sig_t
_signal_nobind(s, a)
        int s;
        sig_t a;
{
    return signal__(s, a, 0);
}
#endif /* __DYNAMIC__ */
#pragma clang diagnostic pop
