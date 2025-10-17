/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/stdio/fwide.c,v 1.2 2008/04/17 22:17:54 jhb Exp $");

#include "namespace.h"
#include <errno.h>
#include <stdio.h>
#include <wchar.h>
#include "un-namespace.h"
#include "libc_private.h"
#include "local.h"

int
fwide(FILE *fp, int mode)
{
	int m;

	FLOCKFILE(fp);
	/* Only change the orientation if the stream is not oriented yet. */
	if (mode != 0 && fp->_orientation == 0)
		fp->_orientation = mode > 0 ? 1 : -1;
	m = fp->_orientation;
	FUNLOCKFILE(fp);

	return (m);
}
