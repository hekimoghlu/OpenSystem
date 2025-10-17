/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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
#include <stdlib.h>			/* 44BSD stdarg.h uses abort() */
#include <stdarg.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <htable.h>
#include <vstring.h>
#include <stringops.h>

/* Global library. */

#include "been_here.h"

#define STR(x)	vstring_str(x)

/* been_here_init - initialize duplicate filter */

BH_TABLE *been_here_init(int limit, int flags)
{
    BH_TABLE *dup_filter;

    dup_filter = (BH_TABLE *) mymalloc(sizeof(*dup_filter));
    dup_filter->limit = limit;
    dup_filter->flags = flags;
    dup_filter->table = htable_create(0);
    return (dup_filter);
}

/* been_here_free - destroy duplicate filter */

void    been_here_free(BH_TABLE *dup_filter)
{
    htable_free(dup_filter->table, (void (*) (void *)) 0);
    myfree((void *) dup_filter);
}

/* been_here - duplicate detector with finer control */

int     been_here(BH_TABLE *dup_filter, const char *fmt,...)
{
    VSTRING *buf = vstring_alloc(100);
    int     status;
    va_list ap;

    /*
     * Construct the string to be checked.
     */
    va_start(ap, fmt);
    vstring_vsprintf(buf, fmt, ap);
    va_end(ap);

    /*
     * Do the duplicate check.
     */
    status = been_here_fixed(dup_filter, vstring_str(buf));

    /*
     * Cleanup.
     */
    vstring_free(buf);
    return (status);
}

/* been_here_fixed - duplicate detector */

int     been_here_fixed(BH_TABLE *dup_filter, const char *string)
{
    VSTRING *folded_string;
    const char *lookup_key;
    int     status;

    /*
     * Special processing: case insensitive lookup.
     */
    if (dup_filter->flags & BH_FLAG_FOLD) {
	folded_string = vstring_alloc(100);
	lookup_key = casefold(folded_string, string);
    } else {
	folded_string = 0;
	lookup_key = string;
    }

    /*
     * Do the duplicate check.
     */
    if (htable_locate(dup_filter->table, lookup_key) != 0) {
	status = 1;
    } else {
	if (dup_filter->limit <= 0
	    || dup_filter->limit > dup_filter->table->used)
	    htable_enter(dup_filter->table, lookup_key, (void *) 0);
	status = 0;
    }
    if (msg_verbose)
	msg_info("been_here: %s: %d", string, status);

    /*
     * Cleanup.
     */
    if (folded_string)
	vstring_free(folded_string);

    return (status);
}

/* been_here_check - query duplicate detector with finer control */

int     been_here_check(BH_TABLE *dup_filter, const char *fmt,...)
{
    VSTRING *buf = vstring_alloc(100);
    int     status;
    va_list ap;

    /*
     * Construct the string to be checked.
     */
    va_start(ap, fmt);
    vstring_vsprintf(buf, fmt, ap);
    va_end(ap);

    /*
     * Do the duplicate check.
     */
    status = been_here_check_fixed(dup_filter, vstring_str(buf));

    /*
     * Cleanup.
     */
    vstring_free(buf);
    return (status);
}

/* been_here_check_fixed - query duplicate detector */

int     been_here_check_fixed(BH_TABLE *dup_filter, const char *string)
{
    VSTRING *folded_string;
    const char *lookup_key;
    int     status;

    /*
     * Special processing: case insensitive lookup.
     */
    if (dup_filter->flags & BH_FLAG_FOLD) {
	folded_string = vstring_alloc(100);
	lookup_key = casefold(folded_string, string);
    } else {
	folded_string = 0;
	lookup_key = string;
    }

    /*
     * Do the duplicate check.
     */
    status = (htable_locate(dup_filter->table, lookup_key) != 0);
    if (msg_verbose)
	msg_info("been_here_check: %s: %d", string, status);

    /*
     * Cleanup.
     */
    if (folded_string)
	vstring_free(folded_string);

    return (status);
}
