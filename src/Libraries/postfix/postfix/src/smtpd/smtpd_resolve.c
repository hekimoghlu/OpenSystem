/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#include <mymalloc.h>
#include <vstring.h>
#include <ctable.h>
#include <stringops.h>
#include <split_at.h>

/* Global library. */

#include <rewrite_clnt.h>
#include <resolve_clnt.h>
#include <mail_proto.h>

/* Application-specific. */

#include <smtpd_resolve.h>

static CTABLE *smtpd_resolve_cache;

#define STR(x) vstring_str(x)
#define SENDER_ADDR_JOIN_CHAR '\n'

/* resolve_pagein - page in an address resolver result */

static void *resolve_pagein(const char *sender_plus_addr, void *unused_context)
{
    const char myname[] = "resolve_pagein";
    static VSTRING *query;
    static VSTRING *junk;
    static VSTRING *sender_buf;
    RESOLVE_REPLY *reply;
    const char *sender;
    const char *addr;

    /*
     * Initialize on the fly.
     */
    if (query == 0) {
	query = vstring_alloc(10);
	junk = vstring_alloc(10);
	sender_buf = vstring_alloc(10);
    }

    /*
     * Initialize.
     */
    reply = (RESOLVE_REPLY *) mymalloc(sizeof(*reply));
    resolve_clnt_init(reply);

    /*
     * Split the sender and address.
     */
    vstring_strcpy(junk, sender_plus_addr);
    sender = STR(junk);
    if ((addr = split_at(STR(junk), SENDER_ADDR_JOIN_CHAR)) == 0)
	msg_panic("%s: bad search key: \"%s\"", myname, sender_plus_addr);

    /*
     * Resolve the address.
     */
    rewrite_clnt_internal(MAIL_ATTR_RWR_LOCAL, sender, sender_buf);
    rewrite_clnt_internal(MAIL_ATTR_RWR_LOCAL, addr, query);
    resolve_clnt_query_from(STR(sender_buf), STR(query), reply);
    vstring_strcpy(junk, STR(reply->recipient));
    casefold(reply->recipient, STR(junk));	/* XXX */

    /*
     * Save the result.
     */
    return ((void *) reply);
}

/* resolve_pageout - page out an address resolver result */

static void resolve_pageout(void *data, void *unused_context)
{
    RESOLVE_REPLY *reply = (RESOLVE_REPLY *) data;

    resolve_clnt_free(reply);
    myfree((void *) reply);
}

/* smtpd_resolve_init - set up global cache */

void    smtpd_resolve_init(int cache_size)
{

    /*
     * Flush a pre-existing cache. The smtpd_check test program requires this
     * after an address class change.
     */
    if (smtpd_resolve_cache)
	ctable_free(smtpd_resolve_cache);

    /*
     * Initialize the resolved address cache. Note: the cache persists across
     * SMTP sessions so we cannot make it dependent on session state.
     */
    smtpd_resolve_cache = ctable_create(cache_size, resolve_pagein,
					resolve_pageout, (void *) 0);
}

/* smtpd_resolve_addr - resolve cached address */

const RESOLVE_REPLY *smtpd_resolve_addr(const char *sender, const char *addr)
{
    static VSTRING *sender_plus_addr_buf;

    /*
     * Initialize on the fly.
     */
    if (sender_plus_addr_buf == 0)
	sender_plus_addr_buf = vstring_alloc(10);

    /*
     * Sanity check.
     */
    if (smtpd_resolve_cache == 0)
	msg_panic("smtpd_resolve_addr: missing initialization");

    /*
     * Reply from the read-through cache.
     */
    vstring_sprintf(sender_plus_addr_buf, "%s%c%s",
		    sender ? sender : RESOLVE_NULL_FROM,
		    SENDER_ADDR_JOIN_CHAR, addr);
    return (const RESOLVE_REPLY *)
	ctable_locate(smtpd_resolve_cache, STR(sender_plus_addr_buf));
}
