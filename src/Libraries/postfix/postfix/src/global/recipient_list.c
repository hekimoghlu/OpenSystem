/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

#include <mymalloc.h>
#include <msg.h>

/* Global library. */

#include "recipient_list.h"

/* recipient_list_init - initialize */

void    recipient_list_init(RECIPIENT_LIST *list, int variant)
{
    list->avail = 1;
    list->len = 0;
    list->info = (RECIPIENT *) mymalloc(sizeof(RECIPIENT));
    list->variant = variant;
}

/* recipient_list_add - add rcpt to list */

void    recipient_list_add(RECIPIENT_LIST *list, long offset,
			           const char *dsn_orcpt, int dsn_notify,
			           const char *orig_rcpt, const char *rcpt)
{
    int     new_avail;

    if (list->len >= list->avail) {
	new_avail = list->avail * 2;
	list->info = (RECIPIENT *)
	    myrealloc((void *) list->info, new_avail * sizeof(RECIPIENT));
	list->avail = new_avail;
    }
    list->info[list->len].orig_addr = mystrdup(orig_rcpt);
    list->info[list->len].address = mystrdup(rcpt);
    list->info[list->len].offset = offset;
    list->info[list->len].dsn_orcpt = mystrdup(dsn_orcpt);
    list->info[list->len].dsn_notify = dsn_notify;
    if (list->variant == RCPT_LIST_INIT_STATUS)
	list->info[list->len].u.status = 0;
    else if (list->variant == RCPT_LIST_INIT_QUEUE)
	list->info[list->len].u.queue = 0;
    else if (list->variant == RCPT_LIST_INIT_ADDR)
	list->info[list->len].u.addr_type = 0;
    list->len++;
}

/* recipient_list_swap - swap recipients between the two recipient lists */

void    recipient_list_swap(RECIPIENT_LIST *a, RECIPIENT_LIST *b)
{
    if (b->variant != a->variant)
	msg_panic("recipient_lists_swap: incompatible recipient list variants");

#define SWAP(t, x)  do { t x = b->x; b->x = a->x ; a->x = x; } while (0)

    SWAP(RECIPIENT *, info);
    SWAP(int, len);
    SWAP(int, avail);
}

/* recipient_list_free - release memory for in-core recipient structure */

void    recipient_list_free(RECIPIENT_LIST *list)
{
    RECIPIENT *rcpt;

    for (rcpt = list->info; rcpt < list->info + list->len; rcpt++) {
	myfree((void *) rcpt->dsn_orcpt);
	myfree((void *) rcpt->orig_addr);
	myfree((void *) rcpt->address);
    }
    myfree((void *) list->info);
}
