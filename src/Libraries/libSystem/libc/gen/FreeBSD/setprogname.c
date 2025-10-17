/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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

#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/setprogname.c,v 1.8 2002/03/29 22:43:41 markm Exp $");

#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/sysctl.h>
#include <crt_externs.h>
#define	__progname	(*_NSGetProgname())

#include "libc_private.h"

void
setprogname(const char *progname)
{
	const char *p;
	char buf[2*MAXCOMLEN+1];
	int mib[2];
	
	p = strrchr(progname, '/');
	if (p != NULL)
		__progname = (char *)(p = p + 1);
	else
		__progname = (char *)(p = progname);

	strlcpy(&buf[0], (char *)(p), sizeof(buf));

	mib[0] = CTL_KERN;
	mib[1] = KERN_PROCNAME;

	/* ignore errors as this is not a hard error */
	sysctl(mib, 2, NULL, NULL, &buf[0], strlen(buf));
}
