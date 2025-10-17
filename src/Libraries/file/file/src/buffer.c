/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
#include "file.h"

#ifndef	lint
FILE_RCSID("@(#)$File: buffer.c,v 1.8 2020/02/16 15:52:49 christos Exp $")
#endif	/* lint */

#include "magic.h"
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>

void
buffer_init(struct buffer *b, int fd, const struct stat *st, const void *data,
    size_t len)
{
	b->fd = fd;
	if (st)
		memcpy(&b->st, st, sizeof(b->st));
	else if (b->fd == -1 || fstat(b->fd, &b->st) == -1)
		memset(&b->st, 0, sizeof(b->st));
	b->fbuf = data;
	b->flen = len;
	b->eoff = 0;
	b->ebuf = NULL;
	b->elen = 0;
}

void
buffer_fini(struct buffer *b)
{
	free(b->ebuf);
	b->ebuf = NULL;
}

int
buffer_fill(const struct buffer *bb)
{
	struct buffer *b = CCAST(struct buffer *, bb);

	if (b->elen != 0)
		return b->elen == FILE_BADSIZE ? -1 : 0;

	if (!S_ISREG(b->st.st_mode))
		goto out;

	b->elen =  CAST(size_t, b->st.st_size) < b->flen ?
	    CAST(size_t, b->st.st_size) : b->flen;
	if ((b->ebuf = malloc(b->elen)) == NULL)
		goto out;

	b->eoff = b->st.st_size - b->elen;
	if (pread(b->fd, b->ebuf, b->elen, b->eoff) == -1) {
		free(b->ebuf);
		b->ebuf = NULL;
		goto out;
	}

	return 0;
out:
	b->elen = FILE_BADSIZE;
	return -1;
}
