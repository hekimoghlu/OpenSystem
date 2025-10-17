/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
#include "common.h"
#include <stdio.h>
#include <err.h>
#include "roken.h"

RCSID("$Id$");

/*
 * Allocate a buffer enough to handle st->st_blksize, if
 * there is such a field, otherwise BUFSIZ.
 */

void *
alloc_buffer (void *oldbuf, size_t *sz, struct stat *st)
{
    size_t new_sz;

    new_sz = BUFSIZ;
#ifdef HAVE_STRUCT_STAT_ST_BLKSIZE
    if (st)
	new_sz = max(BUFSIZ, st->st_blksize);
#endif
    if(new_sz > *sz) {
	if (oldbuf)
	    free (oldbuf);
	oldbuf = malloc (new_sz);
	if (oldbuf == NULL) {
	    warn ("malloc");
	    *sz = 0;
	    return NULL;
	}
	*sz = new_sz;
    }
    return oldbuf;
}

