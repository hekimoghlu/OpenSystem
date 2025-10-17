/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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
static char sccsid[] = "@(#)fwalk.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/fwalk.c,v 1.10 2007/01/09 00:28:06 imp Exp $");

#include <sys/types.h>
#include <machine/atomic.h>
#include <stdio.h>
#include "local.h"
#include "glue.h"

int
_fwalk(int (*function)(FILE *))
{
	FILE *fp;
	int n, ret;
	struct glue *g;

	ret = 0;
	/*
	 * It should be safe to walk the list without locking it;
	 * new nodes are only added to the end and none are ever
	 * removed.
	 *
	 * Avoid locking this list while walking it or else you will
	 * introduce a potential deadlock in [at least] refill.c.
	 */
	for (g = &__sglue; g != NULL; g = g->next)
		for (fp = g->iobs, n = g->niobs; --n >= 0; fp++)
			if ((fp->_flags != 0) && ((fp->_flags & __SIGN) == 0))
				ret |= (*function)(fp);
	return (ret);
}
