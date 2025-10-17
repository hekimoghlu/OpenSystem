/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <dict.h>
#include <mymalloc.h>
#include <stringops.h>

/* Global library. */

#include <cleanup_user.h>
#include <mail_addr_map.h>
#include <quote_822_local.h>

/* Application-specific. */

#include "cleanup.h"

#define STR		vstring_str
#define MAX_RECURSION	10

/* cleanup_map11_external - one-to-one table lookups */

int     cleanup_map11_external(CLEANUP_STATE *state, VSTRING *addr,
			               MAPS *maps, int propagate)
{
    int     count;
    int     expand_to_self;
    ARGV   *new_addr;
    char   *saved_addr;
    int     did_rewrite = 0;

    /*
     * Produce sensible output even in the face of a recoverable error. This
     * simplifies error recovery considerably because we can do delayed error
     * checking in one place, instead of having error handling code all over
     * the place.
     */
    for (count = 0; count < MAX_RECURSION; count++) {
	if ((new_addr = mail_addr_map_opt(maps, STR(addr), propagate,
					  MA_FORM_EXTERNAL, MA_FORM_EXTERNAL,
					  MA_FORM_EXTERNAL)) != 0) {
	    if (new_addr->argc > 1)
		msg_warn("%s: multi-valued %s entry for %s",
			 state->queue_id, maps->title, STR(addr));
	    saved_addr = mystrdup(STR(addr));
	    did_rewrite |= strcmp(new_addr->argv[0], STR(addr));
	    vstring_strcpy(addr, new_addr->argv[0]);
	    expand_to_self = !strcasecmp_utf8(saved_addr, STR(addr));
	    myfree(saved_addr);
	    argv_free(new_addr);
	    if (expand_to_self)
		return (did_rewrite);
	} else if (maps->error != 0) {
	    msg_warn("%s: %s map lookup problem for %s -- "
		     "message not accepted, try again later",
		     state->queue_id, maps->title, STR(addr));
	    state->errs |= CLEANUP_STAT_WRITE;
	    return (did_rewrite);
	} else {
	    return (did_rewrite);
	}
    }
    msg_warn("%s: unreasonable %s map nesting for %s -- "
	     "message not accepted, try again later",
	     state->queue_id, maps->title, STR(addr));
    return (did_rewrite);
}

/* cleanup_map11_tree - rewrite address node */

int     cleanup_map11_tree(CLEANUP_STATE *state, TOK822 *tree,
			           MAPS *maps, int propagate)
{
    VSTRING *temp = vstring_alloc(100);
    int     did_rewrite;

    /*
     * Produce sensible output even in the face of a recoverable error. This
     * simplifies error recovery considerably because we can do delayed error
     * checking in one place, instead of having error handling code all over
     * the place.
     */
    tok822_externalize(temp, tree->head, TOK822_STR_DEFL);
    did_rewrite = cleanup_map11_external(state, temp, maps, propagate);
    tok822_free_tree(tree->head);
    tree->head = tok822_scan(STR(temp), &tree->tail);
    vstring_free(temp);
    return (did_rewrite);
}

/* cleanup_map11_internal - rewrite address internal form */

int     cleanup_map11_internal(CLEANUP_STATE *state, VSTRING *addr,
			               MAPS *maps, int propagate)
{
    VSTRING *temp = vstring_alloc(100);
    int     did_rewrite;

    /*
     * Produce sensible output even in the face of a recoverable error. This
     * simplifies error recovery considerably because we can do delayed error
     * checking in one place, instead of having error handling code all over
     * the place.
     */
    quote_822_local(temp, STR(addr));
    did_rewrite = cleanup_map11_external(state, temp, maps, propagate);
    unquote_822_local(addr, STR(temp));
    vstring_free(temp);
    return (did_rewrite);
}
