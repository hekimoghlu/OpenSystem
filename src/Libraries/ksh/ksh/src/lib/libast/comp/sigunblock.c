/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
#pragma prototyped

#include <ast.h>

#if _lib_sigunblock

NoN(sigunblock)

#else

#include <sig.h>

#ifndef SIG_UNBLOCK
#undef	_lib_sigprocmask
#endif

int
sigunblock(int s)
{
#if _lib_sigprocmask
	int		op;
	sigset_t	mask;

	sigemptyset(&mask);
	if (s)
	{
		sigaddset(&mask, s);
		op = SIG_UNBLOCK;
	}
	else op = SIG_SETMASK;
	return(sigprocmask(op, &mask, NiL));
#else
#if _lib_sigsetmask
	return(sigsetmask(s ? (sigsetmask(0L) & ~sigmask(s)) : 0L));
#else
	NoP(s);
	return(0);
#endif
#endif
}

#endif
