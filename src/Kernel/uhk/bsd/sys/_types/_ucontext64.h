/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
#ifndef _STRUCT_UCONTEXT64

#include <sys/cdefs.h> /* __DARWIN_UNIX03 */

#if __DARWIN_UNIX03
#define _STRUCT_UCONTEXT64      struct __darwin_ucontext64
#else /* !__DARWIN_UNIX03 */
#define _STRUCT_UCONTEXT64      struct ucontext64
#endif /* __DARWIN_UNIX03 */

#include <machine/types.h> /* __darwin_size_t */
#include <machine/_mcontext.h> /* _STRUCT_MCONTEXT */
#include <sys/_types.h> /* __darwin_sigset_t */
#include <sys/_types/_sigaltstack.h> /* _STRUCT_SIGALTSTACK */

_STRUCT_UCONTEXT64
{
	int                     uc_onstack;
	__darwin_sigset_t       uc_sigmask;     /* signal mask used by this context */
	_STRUCT_SIGALTSTACK     uc_stack;       /* stack used by this context */
	_STRUCT_UCONTEXT64      *uc_link;       /* pointer to resuming context */
	__darwin_size_t         uc_mcsize;      /* size of the machine context passed in */
	_STRUCT_MCONTEXT64      *uc_mcontext64; /* pointer to machine specific context */
};
typedef _STRUCT_UCONTEXT64      ucontext64_t;   /* [???] user context */

#endif /* _STRUCT_UCONTEXT64 */
