/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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
/* System library. */

#include "sys_defs.h"
#include <string.h>

/* Utility library. */

#include "vbuf.h"

/* vbuf_unget - implement at least one character pushback */

int     vbuf_unget(VBUF *bp, int ch)
{
    if ((ch & 0xff) != ch || -bp->cnt >= bp->len) {
	bp->flags |= VBUF_FLAG_RD_ERR;	/* This error affects reads! */
	return (VBUF_EOF);
    } else {
	bp->cnt--;
	bp->flags &= ~VBUF_FLAG_EOF;
	return (*--bp->ptr = ch);
    }
}

/* vbuf_get - handle read buffer empty condition */

int     vbuf_get(VBUF *bp)
{
    return (bp->get_ready(bp) ? VBUF_EOF : VBUF_GET(bp));
}

/* vbuf_put - handle write buffer full condition */

int     vbuf_put(VBUF *bp, int ch)
{
    return (bp->put_ready(bp) ? VBUF_EOF : VBUF_PUT(bp, ch));
}

/* vbuf_read - bulk read from buffer */

ssize_t vbuf_read(VBUF *bp, void *buf, ssize_t len)
{
    ssize_t count;
    void   *cp;
    ssize_t n;

#if 0
    for (count = 0; count < len; count++)
	if ((buf[count] = VBUF_GET(bp)) < 0)
	    break;
    return (count);
#else
    for (cp = buf, count = len; count > 0; cp += n, count -= n) {
	if (bp->cnt >= 0 && bp->get_ready(bp))
	    break;
	n = (count < -bp->cnt ? count : -bp->cnt);
	memcpy(cp, bp->ptr, n);
	bp->ptr += n;
	bp->cnt += n;
    }
    return (len - count);
#endif
}

/* vbuf_write - bulk write to buffer */

ssize_t vbuf_write(VBUF *bp, const void *buf, ssize_t len)
{
    ssize_t count;
    const void *cp;
    ssize_t n;

#if 0
    for (count = 0; count < len; count++)
	if (VBUF_PUT(bp, buf[count]) < 0)
	    break;
    return (count);
#else
    for (cp = buf, count = len; count > 0; cp += n, count -= n) {
	if (bp->cnt <= 0 && bp->put_ready(bp) != 0)
	    break;
	n = (count < bp->cnt ? count : bp->cnt);
	memcpy(bp->ptr, cp, n);
	bp->ptr += n;
	bp->cnt -= n;
    }
    return (len - count);
#endif
}
