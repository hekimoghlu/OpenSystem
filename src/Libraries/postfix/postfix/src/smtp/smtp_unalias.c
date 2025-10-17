/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <netdb.h>
#include <string.h>

/* Utility library. */

#include <htable.h>
#include <vstring.h>
#include <msg.h>

/* DNS library. */

#include <dns.h>

/* Application-specific. */

#include "smtp.h"

static int smtp_unalias_flags;

/* smtp_unalias_name - look up the official host or domain name. */

const char *smtp_unalias_name(const char *name)
{
    static HTABLE *cache;
    VSTRING *fqdn;
    char   *result;

    if (*name == '[')
	return (name);

    /*
     * Initialize the cache on the fly. The smtp client is designed to exit
     * after servicing a limited number of requests, so there is no need to
     * prevent the cache from growing too large, or to expire old entries.
     */
    if (cache == 0)
	cache = htable_create(10);

    /*
     * Look up the fqdn. If none is found use the query name instead, so that
     * we won't lose time looking up the same bad name again.
     */
    if ((result = htable_find(cache, name)) == 0) {
	fqdn = vstring_alloc(10);
	if (dns_lookup_l(name, smtp_unalias_flags, (DNS_RR **) 0, fqdn,
			     (VSTRING *) 0, DNS_REQ_FLAG_NONE, T_MX, T_A,
#ifdef HAS_IPV6
			     T_AAAA,
#endif
			     0) != DNS_OK)
	    vstring_strcpy(fqdn, name);
	htable_enter(cache, name, result = vstring_export(fqdn));
    }
    return (result);
}

/* smtp_unalias_addr - rewrite aliases in domain part of address */

VSTRING *smtp_unalias_addr(VSTRING *result, const char *addr)
{
    char   *at;
    const char *fqdn;

    if ((at = strrchr(addr, '@')) == 0 || at[1] == 0) {
	vstring_strcpy(result, addr);
    } else {
	fqdn = smtp_unalias_name(at + 1);
	vstring_strncpy(result, addr, at - addr + 1);
	vstring_strcat(result, fqdn);
    }
    return (result);
}

#ifdef TEST

 /*
  * Test program - read address from stdin, print result on stdout.
  */

#include <vstring_vstream.h>

int     main(int unused_argc, char **unused_argv)
{
    VSTRING *addr = vstring_alloc(10);
    VSTRING *result = vstring_alloc(10);

    smtp_unalias_flags |= RES_DEBUG;

    while (vstring_fgets_nonl(addr, VSTREAM_IN)) {
	smtp_unalias_addr(result, vstring_str(addr));
	vstream_printf("%s -> %s\n", vstring_str(addr), vstring_str(result));
	vstream_fflush(VSTREAM_OUT);
    }
    vstring_free(addr);
    vstring_free(result);
    return (0);
}

#endif
