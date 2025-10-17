/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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
#include <unistd.h>
#include <string.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <vstream.h>
#include <htable.h>
#include <readlline.h>
#include <mymalloc.h>
#include <vstring_vstream.h>
#include <stringops.h>

/* Global library. */

#include <tok822.h>
#include <mail_params.h>
#include <bounce.h>
#include <defer.h>

/* Application-specific. */

#include "local.h"

/* deliver_token_home - expand ~token */

static int deliver_token_home(LOCAL_STATE state, USER_ATTR usr_attr, char *addr)
{
    char   *full_path;
    int     status;

    if (addr[1] != '/') {			/* disallow ~user */
	msg_warn("bad home directory syntax for: %s", addr);
	dsb_simple(state.msg_attr.why, "5.3.5",
		   "mail system configuration error");
	status = bounce_append(BOUNCE_FLAGS(state.request),
			       BOUNCE_ATTR(state.msg_attr));
    } else if (usr_attr.home == 0) {		/* require user context */
	msg_warn("unknown home directory for: %s", addr);
	dsb_simple(state.msg_attr.why, "5.3.5",
		   "mail system configuration error");
	status = bounce_append(BOUNCE_FLAGS(state.request),
			       BOUNCE_ATTR(state.msg_attr));
    } else if (usr_attr.home[0] == '/' && usr_attr.home[1] == 0) {
	status = deliver_file(state, usr_attr, addr + 1);
    } else {					/* expand ~ to home */
	full_path = concatenate(usr_attr.home, addr + 1, (char *) 0);
	status = deliver_file(state, usr_attr, full_path);
	myfree(full_path);
    }
    return (status);
}

/* deliver_token - deliver to expansion of include file or alias */

int     deliver_token(LOCAL_STATE state, USER_ATTR usr_attr, TOK822 *addr)
{
    VSTRING *addr_buf = vstring_alloc(100);
    static char include[] = ":include:";
    int     status;
    char   *path;

    tok822_internalize(addr_buf, addr->head, TOK822_STR_DEFL);
    if (msg_verbose)
	msg_info("deliver_token: %s", STR(addr_buf));

    if (*STR(addr_buf) == '/') {
	status = deliver_file(state, usr_attr, STR(addr_buf));
    } else if (*STR(addr_buf) == '~') {
	status = deliver_token_home(state, usr_attr, STR(addr_buf));
    } else if (*STR(addr_buf) == '|') {
	if ((local_cmd_deliver_mask & state.msg_attr.exp_type) == 0) {
	    dsb_simple(state.msg_attr.why, "5.7.1",
		       "mail to command is restricted");
	    status = bounce_append(BOUNCE_FLAGS(state.request),
				   BOUNCE_ATTR(state.msg_attr));
	} else
	    status = deliver_command(state, usr_attr, STR(addr_buf) + 1);
    } else if (strncasecmp(STR(addr_buf), include, sizeof(include) - 1) == 0) {
	path = STR(addr_buf) + sizeof(include) - 1;
	status = deliver_include(state, usr_attr, path);
    } else {
	status = deliver_resolve_tree(state, usr_attr, addr);
    }
    vstring_free(addr_buf);

    return (status);
}

/* deliver_token_string - tokenize string and deliver */

int     deliver_token_string(LOCAL_STATE state, USER_ATTR usr_attr,
			             char *string, int *addr_count)
{
    TOK822 *tree;
    TOK822 *addr;
    int     status = 0;

    if (msg_verbose)
	msg_info("deliver_token_string: %s", string);

    tree = tok822_parse(string);
    for (addr = tree; addr != 0; addr = addr->next) {
	if (addr->type == TOK822_ADDR) {
	    if (addr_count)
		(*addr_count)++;
	    status |= deliver_token(state, usr_attr, addr);
	}
    }
    tok822_free_tree(tree);
    return (status);
}

/* deliver_token_stream - tokenize stream and deliver */

int     deliver_token_stream(LOCAL_STATE state, USER_ATTR usr_attr,
			             VSTREAM *fp, int *addr_count)
{
    VSTRING *buf = vstring_alloc(100);
    int     status = 0;

    if (msg_verbose)
	msg_info("deliver_token_stream: %s", VSTREAM_PATH(fp));

    while (vstring_fgets_bound(buf, fp, var_line_limit)) {
	if (*STR(buf) != '#') {
	    status = deliver_token_string(state, usr_attr, STR(buf), addr_count);
	    if (status != 0)
		break;
	}
    }
    if (vstream_ferror(fp)) {
	dsb_simple(state.msg_attr.why, "4.3.0",
		   "error reading forwarding file: %m");
	status = defer_append(BOUNCE_FLAGS(state.request),
			      BOUNCE_ATTR(state.msg_attr));
    }
    vstring_free(buf);
    return (status);
}
