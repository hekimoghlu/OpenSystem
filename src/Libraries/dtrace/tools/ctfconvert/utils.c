/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
 * Copyright (c) 1998-2001 by Sun Microsystems, Inc.
 * All rights reserved.
 */

#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>

#include "utils.h"

/*LINTLIBRARY*/

static const char *pname;

#if !defined(__APPLE__)
#pragma init(getpname)
const char *
getpname(void)
{
	const char *p, *q;

	if (pname != NULL)
		return (pname);

	if ((p = getexecname()) != NULL)
		q = strrchr(p, '/');
	else
		q = NULL;

	if (q == NULL)
		pname = p;
	else
		pname = q + 1;

	return (pname);
}
#endif /* __APPLE__ */

__printflike(1, 0)
void
vwarn(const char *format, va_list alist)
{
	int err = errno;

	if (pname != NULL)
		(void) fprintf(stderr, "%s: ", pname);

	(void) vfprintf(stderr, format, alist);

	if (strchr(format, '\n') == NULL)
		(void) fprintf(stderr, ": %s\n", strerror(err));
}

/*PRINTFLIKE1*/
__printflike(1, 2)
void
warn(const char *format, ...)
{
	va_list alist;

	va_start(alist, format);
	vwarn(format, alist);
	va_end(alist);
}

__printflike(1, 0)
void
vdie(const char *format, va_list alist)
{
	vwarn(format, alist);
	exit(E_ERROR);
}

/*PRINTFLIKE1*/
__printflike(1, 2)
void
die(const char *format, ...)
{
	va_list alist;

	va_start(alist, format);
	vdie(format, alist);
	va_end(alist);
}
