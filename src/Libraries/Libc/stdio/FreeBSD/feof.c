/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
static char sccsid[] = "@(#)feof.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/feof.c,v 1.12 2008/05/05 16:03:52 jhb Exp $");

#include "namespace.h"
#include <stdio.h>
#include "un-namespace.h"
#include "libc_private.h"

#undef feof
#undef feof_unlocked

int
feof(FILE *fp)
{
	int	ret;

	FLOCKFILE(fp);
	ret= __sfeof(fp);
	FUNLOCKFILE(fp);
	return (ret);
}

int
feof_unlocked(FILE *fp)
{

	return (__sfeof(fp));
}
