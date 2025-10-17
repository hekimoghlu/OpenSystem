/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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

#include <mymalloc.h>
#include <msg.h>
#include <argv.h>
#include <vstring.h>
#include <dict.h>
#include <stringops.h>

/* Global library. */

#include <mail_params.h>
#include <mail_addr_map.h>
#include <cleanup_user.h>
#include <quote_822_local.h>
#include <been_here.h>

/* Application-specific. */

#include "cleanup.h"

/* cleanup_map1n_internal - one-to-many table lookups */

ARGV   *cleanup_map1n_internal(CLEANUP_STATE *state, const char *addr,
			               MAPS *maps, int propagate)
{
    ARGV   *argv;
    ARGV   *lookup;
    int     count;
    int     i;
    int     arg;
    BH_TABLE *been_here;
    char   *saved_lhs;

    /*
     * Initialize.
     */
    argv = argv_alloc(1);
    argv_add(argv, addr, ARGV_END);
    argv_terminate(argv);
    been_here = been_here_init(0, BH_FLAG_FOLD);

    /*
     * Rewrite the address vector in place. With each map lookup result,
     * split it into separate addresses, then rewrite and flatten each
     * address, and repeat the process. Beware: argv is being changed, so we
     * must index the array explicitly, instead of running along it with a
     * pointer.
     */
#define UPDATE(ptr,new)	do { \
	if (ptr) myfree(ptr); ptr = mystrdup(new); \
    } while (0)
#define STR	vstring_str
#define RETURN(x) do { \
	been_here_free(been_here); return (x); \
    } while (0)
#define UNEXPAND(argv, addr) do { \
	argv_truncate((argv), 0); argv_add((argv), (addr), (char *) 0); \
    } while (0)

    for (arg = 0; arg < argv->argc; arg++) {
	if (argv->argc > var_virt_expan_limit) {
	    msg_warn("%s: unreasonable %s map expansion size for %s -- "
		     "message not accepted, try again later",
		     state->queue_id, maps->title, addr);
	    state->errs |= CLEANUP_STAT_DEFER;
	    UPDATE(state->reason, "4.6.0 Alias expansion error");
	    UNEXPAND(argv, addr);
	    RETURN(argv);
	}
	for (count = 0; /* void */ ; count++) {

	    /*
	     * Don't expand an address that already expanded into itself.
	     */
	    if (been_here_check_fixed(been_here, argv->argv[arg]) != 0)
		break;
	    if (count >= var_virt_recur_limit) {
		msg_warn("%s: unreasonable %s map nesting for %s -- "
			 "message not accepted, try again later",
			 state->queue_id, maps->title, addr);
		state->errs |= CLEANUP_STAT_DEFER;
		UPDATE(state->reason, "4.6.0 Alias expansion error");
		UNEXPAND(argv, addr);
		RETURN(argv);
	    }
	    if ((lookup = mail_addr_map_internal(maps, argv->argv[arg],
						 propagate)) != 0) {
		saved_lhs = mystrdup(argv->argv[arg]);
		for (i = 0; i < lookup->argc; i++) {
		    if (strlen(lookup->argv[i]) > var_virt_addrlen_limit) {
			msg_warn("%s: unreasonable %s result %.300s... -- "
				 "message not accepted, try again later",
			     state->queue_id, maps->title, lookup->argv[i]);
			state->errs |= CLEANUP_STAT_DEFER;
			UPDATE(state->reason, "4.6.0 Alias expansion error");
			UNEXPAND(argv, addr);
			RETURN(argv);
		    }
		    if (i == 0) {
			UPDATE(argv->argv[arg], lookup->argv[i]);
		    } else {
			argv_add(argv, lookup->argv[i], ARGV_END);
			argv_terminate(argv);
		    }

		    /*
		     * Allow an address to expand into itself once.
		     */
		    if (strcasecmp_utf8(saved_lhs, lookup->argv[i]) == 0)
			been_here_fixed(been_here, saved_lhs);
		}
		myfree(saved_lhs);
		argv_free(lookup);
	    } else if (maps->error != 0) {
		msg_warn("%s: %s map lookup problem for %s -- "
			 "message not accepted, try again later",
			 state->queue_id, maps->title, addr);
		state->errs |= CLEANUP_STAT_WRITE;
		UPDATE(state->reason, "4.6.0 Alias expansion error");
		UNEXPAND(argv, addr);
		RETURN(argv);
	    } else {
		break;
	    }
	}
    }
    RETURN(argv);
}
