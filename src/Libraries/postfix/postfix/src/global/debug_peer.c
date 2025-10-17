/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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

#include <msg.h>

/* Global library. */

#include <mail_params.h>
#include <namadr_list.h>
#include <debug_peer.h>
#include <match_parent_style.h>

/* Application-specific. */

#define UNUSED_SAVED_LEVEL	(-1)

static NAMADR_LIST *debug_peer_list;
static int saved_level = UNUSED_SAVED_LEVEL;

/* debug_peer_init - initialize */

void    debug_peer_init(void)
{
    const char *myname = "debug_peer_init";

    /*
     * Sanity check.
     */
    if (debug_peer_list)
	msg_panic("%s: repeated call", myname);
    if (var_debug_peer_list == 0)
	msg_panic("%s: uninitialized %s", myname, VAR_DEBUG_PEER_LIST);
    if (var_debug_peer_level <= 0)
	msg_fatal("%s: %s <= 0", myname, VAR_DEBUG_PEER_LEVEL);

    /*
     * Finally.
     */
    if (*var_debug_peer_list)
	debug_peer_list =
	    namadr_list_init(VAR_DEBUG_PEER_LIST, MATCH_FLAG_RETURN
			     | match_parent_style(VAR_DEBUG_PEER_LIST),
			     var_debug_peer_list);
}

/* debug_peer_check - see if this peer needs verbose logging */

int     debug_peer_check(const char *name, const char *addr)
{

    /*
     * Crank up the noise when this peer is listed.
     */
    if (debug_peer_list != 0
	&& saved_level == UNUSED_SAVED_LEVEL
	&& namadr_list_match(debug_peer_list, name, addr) != 0) {
	saved_level = msg_verbose;
	msg_verbose += var_debug_peer_level;
	return (1);
    }
    return (0);
}

/* debug_peer_restore - restore logging level */

void    debug_peer_restore(void)
{
    if (saved_level != UNUSED_SAVED_LEVEL) {
	msg_verbose = saved_level;
	saved_level = UNUSED_SAVED_LEVEL;
    }
}
