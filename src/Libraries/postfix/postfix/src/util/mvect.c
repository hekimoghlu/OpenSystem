/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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

#include <sys_defs.h>

/* Utility library. */

#include "mymalloc.h"
#include "mvect.h"

/* mvect_alloc - allocate memory vector */

char   *mvect_alloc(MVECT *vect, ssize_t elsize, ssize_t nelm,
               void (*init_fn) (char *, ssize_t), void (*wipe_fn) (char *, ssize_t))
{
    vect->init_fn = init_fn;
    vect->wipe_fn = wipe_fn;
    vect->nelm = 0;
    vect->ptr = mymalloc(elsize * nelm);
    vect->nelm = nelm;
    vect->elsize = elsize;
    if (vect->init_fn)
	vect->init_fn(vect->ptr, vect->nelm);
    return (vect->ptr);
}

/* mvect_realloc - adjust memory vector allocation */

char   *mvect_realloc(MVECT *vect, ssize_t nelm)
{
    ssize_t old_len = vect->nelm;
    ssize_t incr = nelm - old_len;
    ssize_t new_nelm;

    if (incr > 0) {
	if (incr < old_len)
	    incr = old_len;
	new_nelm = vect->nelm + incr;
	vect->ptr = myrealloc(vect->ptr, vect->elsize * new_nelm);
	vect->nelm = new_nelm;
	if (vect->init_fn)
	    vect->init_fn(vect->ptr + old_len * vect->elsize, incr);
    }
    return (vect->ptr);
}

/* mvect_free - release memory vector storage */

char   *mvect_free(MVECT *vect)
{
    if (vect->wipe_fn)
	vect->wipe_fn(vect->ptr, vect->nelm);
    myfree(vect->ptr);
    return (0);
}
