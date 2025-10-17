/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
static char sccsid[] = "@(#)putchar.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/putchar.c,v 1.16 2008/05/05 16:03:52 jhb Exp $");

#include "namespace.h"
#include <stdio.h>
#include "un-namespace.h"
#include "local.h"
#include "libc_private.h"

#undef putchar
#undef putchar_unlocked

/*
 * A subroutine version of the macro putchar
 */
int
putchar(c)
	int c;
{
	int retval;
	FILE *so = stdout;

	FLOCKFILE(so);
	/* Orientation set by __sputc() when buffer is full. */
	/* ORIENT(so, -1); */
	retval = __sputc(c, so);
	FUNLOCKFILE(so);
	return (retval);
}

int
putchar_unlocked(int ch)
{

	return (__sputc(ch, stdout));
}
