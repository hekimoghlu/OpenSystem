/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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

#include <mail_proto.h>

/* Application-specific. */

#include <smtpd.h>

/* smtpd_xforward_init - initialize xforward attributes */

void    smtpd_xforward_init(SMTPD_STATE *state)
{
    state->xforward.flags = 0;
    state->xforward.name = 0;
    state->xforward.addr = 0;
    state->xforward.port = 0;
    state->xforward.namaddr = 0;
    state->xforward.protocol = 0;
    state->xforward.helo_name = 0;
    state->xforward.ident = 0;
    state->xforward.domain = 0;
}

/* smtpd_xforward_preset - set xforward attributes to "unknown" */

void    smtpd_xforward_preset(SMTPD_STATE *state)
{

    /*
     * Sanity checks.
     */
    if (state->xforward.flags)
	msg_panic("smtpd_xforward_preset: bad flags: 0x%x",
		  state->xforward.flags);

    /*
     * This is a temporary solution. Unknown forwarded attributes get the
     * same values as unknown normal attributes, so that we don't break
     * assumptions in pre-existing code.
     */
    state->xforward.flags = SMTPD_STATE_XFORWARD_INIT;
    state->xforward.name = mystrdup(CLIENT_NAME_UNKNOWN);
    state->xforward.addr = mystrdup(CLIENT_ADDR_UNKNOWN);
    state->xforward.port = mystrdup(CLIENT_PORT_UNKNOWN);
    state->xforward.namaddr = mystrdup(CLIENT_NAMADDR_UNKNOWN);
    state->xforward.rfc_addr = mystrdup(CLIENT_ADDR_UNKNOWN);
    /* Leave helo at zero. */
    state->xforward.protocol = mystrdup(CLIENT_PROTO_UNKNOWN);
    /* Leave ident at zero. */
    /* Leave domain context at zero. */
}

/* smtpd_xforward_reset - reset xforward attributes */

void    smtpd_xforward_reset(SMTPD_STATE *state)
{
#define FREE_AND_WIPE(s) { if (s) myfree(s); s = 0; }

    state->xforward.flags = 0;
    FREE_AND_WIPE(state->xforward.name);
    FREE_AND_WIPE(state->xforward.addr);
    FREE_AND_WIPE(state->xforward.port);
    FREE_AND_WIPE(state->xforward.namaddr);
    FREE_AND_WIPE(state->xforward.rfc_addr);
    FREE_AND_WIPE(state->xforward.protocol);
    FREE_AND_WIPE(state->xforward.helo_name);
    FREE_AND_WIPE(state->xforward.ident);
    FREE_AND_WIPE(state->xforward.domain);
}
