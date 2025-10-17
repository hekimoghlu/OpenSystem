/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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

#include <vstring.h>
#include <msg.h>

/* Global library. */

#include "resolve_clnt.h"
#include "tok822.h"

/* tok822_resolve - address rewriting interface */

void    tok822_resolve_from(const char *sender, TOK822 *addr,
			            RESOLVE_REPLY *reply)
{
    VSTRING *intern_form = vstring_alloc(100);

    if (addr->type != TOK822_ADDR)
	msg_panic("tok822_resolve: non-address token type: %d", addr->type);

    /*
     * Internalize the token tree and ship it to the resolve service.
     * Shipping string forms is much simpler than shipping parse trees.
     */
    tok822_internalize(intern_form, addr->head, TOK822_STR_DEFL);
    resolve_clnt_query_from(sender, vstring_str(intern_form), reply);
    if (msg_verbose)
	msg_info("tok822_resolve: from=%s addr=%s -> chan=%s, host=%s, rcpt=%s",
		 sender,
		 vstring_str(intern_form), vstring_str(reply->transport),
		 vstring_str(reply->nexthop), vstring_str(reply->recipient));

    vstring_free(intern_form);
}
