/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#include <config.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "buffer.h"

void
ct_buf_init(ct_buf_t *bp, void *mem, size_t len)
{
	memset(bp, 0, sizeof(*bp));
	bp->base = (unsigned char *) mem;
	bp->size = len;
}

void
ct_buf_set(ct_buf_t *bp, void *mem, size_t len)
{
	ct_buf_init(bp, mem, len);
	bp->tail = len;
}

int
ct_buf_get(ct_buf_t *bp, void *mem, size_t len)
{
	if (len > bp->tail - bp->head)
		return -1;
	if (mem)
		memcpy(mem, bp->base + bp->head, len);
	bp->head += len;
	return len;
}

int
ct_buf_put(ct_buf_t *bp, const void *mem, size_t len)
{
	if (len > bp->size - bp->tail) {
		bp->overrun = 1;
		return -1;
	}
	if (mem)
		memcpy(bp->base + bp->tail, mem, len);
	bp->tail += len;
	return len;
}

int
ct_buf_putc(ct_buf_t *bp, int byte)
{
	unsigned char	c = byte;

	return ct_buf_put(bp, &c, 1);
}

unsigned int
ct_buf_avail(ct_buf_t *bp)
{
	return bp->tail - bp->head;
}

void *
ct_buf_head(ct_buf_t *bp)
{
	return bp->base + bp->head;
}

