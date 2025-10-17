/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
static char sccsid[] = "@(#)getlogin.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/getlogin.c,v 1.11 2009/12/05 19:04:21 ed Exp $");

#include <sys/param.h>
#include <errno.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "namespace.h"
#include <pthread.h>
#include "un-namespace.h"

#include "libc_private.h"

extern int __getlogin(char *, int);

__private_extern__ pthread_mutex_t __logname_mutex = PTHREAD_MUTEX_INITIALIZER;
__private_extern__ char *__logname = NULL;

static char *
getlogin_basic(int *status)
{
	if (__logname == NULL) {
		__logname = calloc(1, MAXLOGNAME);
		if (__logname == NULL) {
			*status = ENOMEM;
			return (NULL);
		}
	}

	if (__logname[0] == 0) {
		if (__getlogin(__logname, MAXLOGNAME) < 0) {
			*status = errno;
			return (NULL);
		}
	}
	*status = 0;
	return (*__logname ? __logname : NULL);
}

char *
getlogin(void)
{
	char	*result;
	int	status;

	pthread_mutex_lock(&__logname_mutex);
	result = getlogin_basic(&status);
	pthread_mutex_unlock(&__logname_mutex);
	return (result);
}

int
getlogin_r(char *logname, size_t namelen)
{
	char	*result;
	int	status;

	pthread_mutex_lock(&__logname_mutex);
	result = getlogin_basic(&status);
	if (status == 0) {
		if (strlcpy(logname, __logname, namelen) > namelen) {
			status = ERANGE;
		}
	}
	pthread_mutex_unlock(&__logname_mutex);

	return (status);
}
