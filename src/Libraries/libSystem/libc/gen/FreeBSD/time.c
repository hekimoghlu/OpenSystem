/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
static char sccsid[] = "@(#)time.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/time.c,v 1.5 2007/01/09 00:27:55 imp Exp $");

#include <sys/types.h>
#include <sys/time.h>
#include <fenv.h>

time_t
time(t)
	time_t *t;
{
	struct timeval tt;
	time_t retval;
#ifdef FE_DFL_ENV
	fenv_t fenv;
#endif /* FE_DFL_ENV */

#ifdef FE_DFL_ENV
	fegetenv(&fenv); /* 3965505 - need to preserve floating point enviroment */
#endif /* FE_DFL_ENV */
	if (gettimeofday(&tt, (struct timezone *)0) < 0)
		retval = -1;
	else
		retval = tt.tv_sec;
	if (t != NULL)
		*t = retval;
#ifdef FE_DFL_ENV
	fesetenv(&fenv);
#endif /* FE_DFL_ENV */
	return (retval);
}
