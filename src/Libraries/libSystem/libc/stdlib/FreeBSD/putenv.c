/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
static char sccsid[] = "@(#)putenv.c	8.2 (Berkeley) 3/27/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdlib/putenv.c,v 1.6 2007/05/01 16:02:41 ache Exp $");

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <db.h>
#include <crt_externs.h>
#include <errno.h> 

extern struct owned_ptr *__env_owned;

#ifdef LEGACY_CRT1_ENVIRON
extern char **_saved_environ;
#endif /* LEGACY_CRT1_ENVIRON */

__private_extern__ int __init__env_owned_locked(int);
__private_extern__ int __setenv_locked(const char *, const char *, int, int, char ***, struct owned_ptr *);
__private_extern__ void __environ_lock(void);
__private_extern__ void __environ_unlock(void);

#ifndef BUILDING_VARIANT
/*
 * _putenvp -- SPI using an arbitrary pointer to string array (the array must
 * have been created with malloc) and an env state, created by _allocenvstate().
 *	Returns ptr to value associated with name, if any, else NULL.
 */
int
_putenvp(char *str, char ***envp, void *state)
{
	__environ_lock();
	if (__init__env_owned_locked(1)) {
		__environ_unlock();
		return (-1);
	}
	int ret = __setenv_locked(str, NULL, 1, 0, envp,
			(state ? (struct owned_ptr *)state : __env_owned));
	__environ_unlock();
	return ret;
}
#endif /* BUILDING_VARIANT */

int
putenv(str)
	char *str;
{
	int ret;
	int copy;

#if __DARWIN_UNIX03
	if (str == NULL || *str == 0 || index(str, '=') == NULL) {
		errno = EINVAL;
		return (-1);
	}
#else /* !__DARWIN_UNIX03 */
	if (index(str, '=') == NULL)
		return (-1);
#endif /* __DARWIN_UNIX03 */

#if __DARWIN_UNIX03
	copy = 0;
#else /* !__DARWIN_UNIX03 */
	copy = -1;
#endif /* __DARWIN_UNIX03 */

	__environ_lock();
	if (__init__env_owned_locked(1)) {
		__environ_unlock();
		return (-1);
	}
	ret = __setenv_locked(str, NULL, 1, copy, _NSGetEnviron(), __env_owned);
#ifdef LEGACY_CRT1_ENVIRON
	_saved_environ = *_NSGetEnviron();
#endif /* LEGACY_CRT1_ENVIRON */
	__environ_unlock();
	return ret;
}
