/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
/*
 * Copyright (c) 2001 Daniel M. Eischen <deischen@freebsd.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Neither the name of the author nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#define _XOPEN_SOURCE 600L
#include <ucontext.h>
#include <errno.h>
#include <TargetConditionals.h>

/* This is a macro to capture all the code added in here that is purely to make
 * conformance tests pass and seems to have no functional reason nor is it
 * required by the standard */
#define CONFORMANCE_SPECIFIC_HACK 1

#if ((TARGET_OS_OSX || TARGET_OS_DRIVERKIT) && (defined(__LP64__) || defined(__i386__)))

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <sys/param.h>
#include <sys/signal.h>
#include <stddef.h>

#define uc_flags uc_onstack
#define UCF_SWAPPED 0x80000000

int
swapcontext(ucontext_t *oucp, const ucontext_t *ucp)
{
	int ret;
	if ((oucp == NULL) || (ucp == NULL)) {
		errno = EINVAL;
		return -1;
	}

	oucp->uc_flags &= ~UCF_SWAPPED;

#if CONFORMANCE_SPECIFIC_HACK
	// getcontext overwrites uc_link so we save it and restore it
	ucontext_t *next_context = oucp->uc_link;
	ret = getcontext(oucp);
	oucp->uc_link = next_context;
#endif

	if ((ret == 0) && !(oucp->uc_flags & UCF_SWAPPED)) {
		oucp->uc_flags |= UCF_SWAPPED;
		/* In the future, when someone calls setcontext(oucp), that will return
		 * us to the getcontext call above with ret = 0. However, because we
		 * just flipped the UCF_SWAPPED bit, we will not call setcontext again
		 * and will return. */
		ret = setcontext(ucp);
	}

	asm(""); // Prevent tailcall <rdar://problem/12581792>
	return (ret);
}

#else /* TARGET_OS_OSX || TARGET_OS_DRIVERKIT */

int
swapcontext(ucontext_t *oucp, const ucontext_t *ucp)
{
	errno = ENOTSUP;
	return -1;
}

#endif /* TARGET_OS_OSX || TARGET_OS_DRIVERKIT */
