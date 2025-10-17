/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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

#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <setjmp.h>
#if __has_feature(ptrauth_calls) && !defined(__OPEN_SOURCE)
#include <ptrauth.h>
#endif

#include <darwintest.h>

static char *heap;
static volatile int pass;
static sigjmp_buf jbuf;

static void __dead2
action(int signo, struct __siginfo *info, void *uap __attribute__((unused)))
{
	if (info) {
		pass = (signo == SIGBUS && info->si_addr == heap);
	}
	siglongjmp(jbuf, 0);
}

T_DECL(nxheap, "Non-executable heap", T_META_CHECK_LEAKS(false), T_META_ASROOT(true))
{
	struct sigaction sa = {
		.__sigaction_u.__sa_sigaction = action,
		.sa_flags = SA_SIGINFO,
	};

	T_ASSERT_POSIX_ZERO(sigaction(SIGBUS, &sa, NULL), NULL);

	if (sigsetjmp(jbuf, 0)) {
		T_PASS("SIGBUS");
		T_END;
	}

	T_QUIET; T_ASSERT_NOTNULL((heap = malloc(1)), NULL);

	*heap = (char)0xc3; // retq
#if __has_feature(ptrauth_calls) && !defined(__OPEN_SOURCE)
	heap = ptrauth_sign_unauthenticated(heap, ptrauth_key_function_pointer, 0);
#endif
	((void (*)(void))heap)(); // call *%eax

	T_FAIL("SIGBUS");
}
