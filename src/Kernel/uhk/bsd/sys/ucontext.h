/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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
#ifndef _SYS_UCONTEXT_H_
#define _SYS_UCONTEXT_H_

#include <sys/cdefs.h>
#include <sys/_types.h>

#include <machine/_mcontext.h>
#include <sys/_types/_ucontext.h>

#include <sys/_types/_sigset_t.h>

#ifdef KERNEL
#include <machine/types.h>      /* user_addr_t, user_size_t */

/* kernel representation of struct ucontext64 for 64 bit processes */
typedef struct user_ucontext64 {
	int                             uc_onstack;
	sigset_t                        uc_sigmask;     /* signal mask */
	struct user64_sigaltstack       uc_stack;       /* stack */
	user_addr_t                     uc_link;        /* ucontext pointer */
	user_size_t                     uc_mcsize;      /* mcontext size */
	user_addr_t                     uc_mcontext64;  /* machine context */
} user_ucontext64_t;

typedef struct user_ucontext32 {
	int                             uc_onstack;
	sigset_t                        uc_sigmask;     /* signal mask */
	struct user32_sigaltstack       uc_stack;       /* stack */
	user32_addr_t                   uc_link;        /* ucontext pointer */
	user32_size_t                   uc_mcsize;      /* mcontext size */
	user32_addr_t                   uc_mcontext;    /* machine context */
} user_ucontext32_t;

#endif  /* KERNEL */

#endif /* _SYS_UCONTEXT_H_ */
