/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 26, 2024.
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
/*	$OpenBSD: explicit_bzero.c,v 1.1 2014/01/22 21:06:45 tedu Exp $ */
/*
 * Public domain.
 * Written by Ted Unangst
 */

#include "includes.h"

#include <string.h>

/*
 * explicit_bzero - don't let the compiler optimize away bzero
 */

#ifndef HAVE_EXPLICIT_BZERO

#ifdef HAVE_EXPLICIT_MEMSET

void
explicit_bzero(void *p, size_t n)
{
	(void)explicit_memset(p, 0, n);
}

#elif defined(HAVE_MEMSET_S)

void
explicit_bzero(void *p, size_t n)
{
	if (n == 0)
		return;
	(void)memset_s(p, n, 0, n);
}

#else /* HAVE_MEMSET_S */

/*
 * Indirect bzero through a volatile pointer to hopefully avoid
 * dead-store optimisation eliminating the call.
 */
static void (* volatile ssh_bzero)(void *, size_t) = bzero;

void
explicit_bzero(void *p, size_t n)
{
	if (n == 0)
		return;
	/*
	 * clang -fsanitize=memory needs to intercept memset-like functions
	 * to correctly detect memory initialisation. Make sure one is called
	 * directly since our indirection trick above successfully confuses it.
	 */
#if defined(__has_feature)
# if __has_feature(memory_sanitizer)
	memset(p, 0, n);
# endif
#endif

	ssh_bzero(p, n);
}

#endif /* HAVE_MEMSET_S */

#endif /* HAVE_EXPLICIT_BZERO */
