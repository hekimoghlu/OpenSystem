/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
static char sccsid[] = "@(#)getenv.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdlib/getenv.c,v 1.8 2007/05/01 16:02:41 ache Exp $");

#include <os/lock_private.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <crt_externs.h>

#include "libc_private.h"

__private_extern__ char *__findenv_locked(const char *, int *, char **);

/*
 * __findenv_locked --
 *	Returns pointer to value associated with name, if any, else NULL.
 *	Sets offset to be the offset of the name/value combination in the
 *	environmental array, for use by setenv(3) and unsetenv(3).
 *	Explicitly removes '=' in argument name.
 *
 *	This routine *should* be a static; don't use it.
 */
__private_extern__ char *
__findenv_locked(const char *name, int *offset, char **environ)
{
	int len, i;
	const char *np;
	char **p, *cp;

	if (name == NULL || environ == NULL)
		return (NULL);
	for (np = name; *np && *np != '='; ++np)
		continue;
	len = np - name;
	for (p = environ; (cp = *p) != NULL; ++p) {
		for (np = name, i = len; i && *cp; i--)
			if (*cp++ != *np++)
				break;
		if (i == 0 && *cp++ == '=') {
			*offset = p - environ;
			return (cp);
		}
	}
	return (NULL);
}

static os_unfair_lock __environ_lock_obj = OS_UNFAIR_LOCK_INIT;
void
environ_lock_np(void)
{
	os_unfair_lock_lock_with_options(
			&__environ_lock_obj, OS_UNFAIR_LOCK_DATA_SYNCHRONIZATION);
}
void
environ_unlock_np(void)
{
	os_unfair_lock_unlock(&__environ_lock_obj);
}
__private_extern__ void
__environ_lock_fork_child(void)
{
	__environ_lock_obj = OS_UNFAIR_LOCK_INIT;
}

/*
 * _getenvp -- SPI using an arbitrary pointer to string array (the array must
 * have been created with malloc) and an env state, created by _allocenvstate().
 *	Returns ptr to value associated with name, if any, else NULL.
 */
char *
_getenvp(const char *name, char ***envp, void *state __unused)
{
	// envp is passed as an argument, so the lock is not protecting everything
	int offset;
	environ_lock_np();
	char *result = (__findenv_locked(name, &offset, *envp));
	environ_unlock_np();
	return result;
}

/*
 * getenv --
 *	Returns ptr to value associated with name, if any, else NULL.
 */
char *
getenv(const char *name)
{
	int offset;
	environ_lock_np();
	char *result = __findenv_locked(name, &offset, *_NSGetEnviron());
	environ_unlock_np();
	return result;
}
