/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
/*
  * System library.
  */
#include <sys_defs.h>
#include <ctype.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

 /*
  * Utility library.
  */
#include <msg.h>
#include <vstring.h>
#include <myaddrinfo.h>

 /*
  * Global library.
  */
#include <maps.h>

 /*
  * DNS library.
  */
#define LIBDNS_INTERNAL
#include <dns.h>

 /*
  * Application-specific.
  */
MAPS   *dns_rr_filter_maps;

static DNS_RR dns_rr_filter_error[1];

#define STR vstring_str

/* dns_rr_filter_compile - compile dns result filter */

void    dns_rr_filter_compile(const char *title, const char *map_names)
{
    if (dns_rr_filter_maps != 0)
	maps_free(dns_rr_filter_maps);
    dns_rr_filter_maps = maps_create(title, map_names,
				     DICT_FLAG_LOCK | DICT_FLAG_FOLD_FIX);
}

/* dns_rr_action - execute action from filter map */

static DNS_RR *dns_rr_action(const char *cmd, DNS_RR *rr, const char *rr_text)
{
    const char *cmd_args = cmd + strcspn(cmd, " \t");
    int     cmd_len = cmd_args - cmd;

    while (*cmd_args && ISSPACE(*cmd_args))
	cmd_args++;

#define STREQUAL(x,y,l) (strncasecmp((x), (y), (l)) == 0 && (y)[l] == 0)

    if (STREQUAL(cmd, "IGNORE", cmd_len)) {
	msg_info("ignoring DNS RR: %s", rr_text);
	return (0);
    } else {
	msg_warn("%s: unknown DNS filter action: \"%s\"", 
		 dns_rr_filter_maps->title, cmd);
	return (dns_rr_filter_error);
    }
    return (rr);
}

/* dns_rr_filter_execute - filter DNS lookup result */

int     dns_rr_filter_execute(DNS_RR **rrlist)
{
    static VSTRING *buf = 0;
    DNS_RR **rrp;
    DNS_RR *rr;
    const char *map_res;
    DNS_RR *act_res;

    /*
     * Convert the resource record to string form, then search the maps for a
     * matching action.
     */
    if (buf == 0)
	buf = vstring_alloc(100);
    for (rrp = rrlist; (rr = *rrp) != 0; /* see below */ ) {
	map_res = maps_find(dns_rr_filter_maps, dns_strrecord(buf, rr),
			    DICT_FLAG_NONE);
	if (map_res != 0) {
	    if ((act_res = dns_rr_action(map_res, rr, STR(buf))) == 0) {
		*rrp = rr->next;		/* do not advance in the list */
		rr->next = 0;
		dns_rr_free(rr);
		continue;
	    } else if (act_res == dns_rr_filter_error) {
		return (-1);
	    }
	} else if (dns_rr_filter_maps->error) {
	    return (-1);
	}
	rrp = &(rr->next);			/* do advance in the list */
    }
    return (0);
}
