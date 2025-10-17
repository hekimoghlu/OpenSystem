/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
#if 0
#ifndef lint
static char sccsid[] = "@(#)nlist.c	8.4 (Berkeley) 4/2/94";
#endif /* not lint */
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/bin/ps/nlist.c,v 1.21 2004/04/06 20:06:49 markm Exp $");

#include <sys/types.h>
#include <sys/sysctl.h>
#ifdef __APPLE__
#include <sys/resource.h>
#endif

#include <stddef.h>

#include "ps.h"

fixpt_t	ccpu;				/* kernel _ccpu variable */
int	nlistread;			/* if nlist already read. */
#ifdef __APPLE__
uint64_t	mempages;		/* number of pages of phys. memory */
#else
unsigned long	mempages;		/* number of pages of phys. memory */
#endif
int	fscale;				/* kernel _fscale variable */

int
donlist(void)
{
#ifdef __APPLE__
	int mib[2];
#endif
	size_t oldlen;

#ifdef __APPLE__
	mib[0] = CTL_HW;
	mib[1] = HW_MEMSIZE;
	oldlen = sizeof(mempages);
	if (sysctl(mib, 2, &mempages, &oldlen, NULL, 0) == -1)
		return (1);
	fscale = 100;
#else
	oldlen = sizeof(ccpu);
	if (sysctlbyname("kern.ccpu", &ccpu, &oldlen, NULL, 0) == -1)
		return (1);
	oldlen = sizeof(fscale);
	if (sysctlbyname("kern.fscale", &fscale, &oldlen, NULL, 0) == -1)
		return (1);
	oldlen = sizeof(mempages);
	if (sysctlbyname("hw.availpages", &mempages, &oldlen, NULL, 0) == -1)
		return (1);
#endif
	nlistread = 1;
	return (0);
}
