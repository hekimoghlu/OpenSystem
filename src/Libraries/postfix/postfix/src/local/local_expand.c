/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 20, 2023.
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

#include <vstring.h>
#include <mac_expand.h>

/* Global library */

#include <mail_params.h>

/* Application-specific. */

#include "local.h"

typedef struct {
    LOCAL_STATE *state;
    USER_ATTR *usr_attr;
    int     status;
} LOCAL_EXP;

/* local_expand_lookup - mac_expand() lookup routine */

static const char *local_expand_lookup(const char *name, int mode, void *ptr)
{
    LOCAL_EXP *local = (LOCAL_EXP *) ptr;
    static char rcpt_delim[2];

#define STREQ(x,y) (*(x) == *(y) && strcmp((x), (y)) == 0)

    if (STREQ(name, "user")) {
	return (local->state->msg_attr.user);
    } else if (STREQ(name, "home")) {
	return (local->usr_attr->home);
    } else if (STREQ(name, "shell")) {
	return (local->usr_attr->shell);
    } else if (STREQ(name, "domain")) {
	return (local->state->msg_attr.domain);
    } else if (STREQ(name, "local")) {
	return (local->state->msg_attr.local);
    } else if (STREQ(name, "mailbox")) {
	return (local->state->msg_attr.local);
    } else if (STREQ(name, "recipient")) {
	return (local->state->msg_attr.rcpt.address);
    } else if (STREQ(name, "extension")) {
	if (mode == MAC_EXP_MODE_USE)
	    local->status |= LOCAL_EXP_EXTENSION_MATCHED;
	return (local->state->msg_attr.extension);
    } else if (STREQ(name, "recipient_delimiter")) {
	rcpt_delim[0] =
	    local->state->msg_attr.local[strlen(local->state->msg_attr.user)];
	rcpt_delim[1] = 0;
	return (rcpt_delim[0] ? rcpt_delim : 0);
#if 0
    } else if (STREQ(name, "client_hostname")) {
	return (local->state->msg_attr.request->client_name);
    } else if (STREQ(name, "client_address")) {
	return (local->state->msg_attr.request->client_addr);
    } else if (STREQ(name, "client_protocol")) {
	return (local->state->msg_attr.request->client_proto);
    } else if (STREQ(name, "client_helo")) {
	return (local->state->msg_attr.request->client_helo);
    } else if (STREQ(name, "sasl_method")) {
	return (local->state->msg_attr.request->sasl_method);
    } else if (STREQ(name, "sasl_sender")) {
	return (local->state->msg_attr.request->sasl_sender);
    } else if (STREQ(name, "sasl_username")) {
	return (local->state->msg_attr.request->sasl_username);
#endif
    } else {
	return (0);
    }
}

/* local_expand - expand message delivery attributes */

int     local_expand(VSTRING *result, const char *pattern,
	        LOCAL_STATE *state, USER_ATTR *usr_attr, const char *filter)
{
    LOCAL_EXP local;
    int     expand_status;

    local.state = state;
    local.usr_attr = usr_attr;
    local.status = 0;
    expand_status = mac_expand(result, pattern, MAC_EXP_FLAG_NONE,
			       filter, local_expand_lookup, (void *) &local);
    return (local.status | expand_status);
}
