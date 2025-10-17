/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 20, 2023.
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
 * Copyright (c) 1997 Todd C. Miller <millert@openbsd.org>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include "local.h"

#define	INITIAL_SIZE	128

int
vasprintf(char **str, const char *fmt, __va_list ap)
{
	int ret;
	FILE f;
	struct __sfileext fext;
	const int pgsz = getpagesize();

	_FILEEXT_SETUP(&f, &fext);
	f._file = -1;
	f._flags = __SWR | __SSTR | __SALC;
	f._bf._base = f._p = malloc(INITIAL_SIZE);
	if (f._bf._base == NULL)
		goto err;
	f._bf._size = f._w = INITIAL_SIZE - 1;	/* leave room for the NUL */
	ret = __vfprintf(&f, fmt, ap);
	if (ret == -1)
		goto err;
	*f._p = '\0';
	if (ret + 1 > INITIAL_SIZE && ret + 1 < pgsz / 2) {
		/* midsize allocations can try to conserve memory */
		unsigned char *_base = recallocarray(f._bf._base,
		    f._bf._size + 1, ret + 1, 1);

		if (_base == NULL)
			goto err;
		*str = (char *)_base;
	} else
		*str = (char *)f._bf._base;
	return (ret);

err:
	free(f._bf._base);
	f._bf._base = NULL;
	*str = NULL;
	errno = ENOMEM;
	return (-1);
}
DEF_WEAK(vasprintf);

