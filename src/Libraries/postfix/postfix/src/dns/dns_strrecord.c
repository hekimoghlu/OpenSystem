/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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
#include <string.h>			/* memcpy */

/* Utility library. */

#include <vstring.h>
#include <msg.h>

/* DNS library. */

#include <dns.h>

/* dns_strrecord - format resource record as generic string */

char   *dns_strrecord(VSTRING *buf, DNS_RR *rr)
{
    const char myname[] = "dns_strrecord";
    MAI_HOSTADDR_STR host;
    UINT32_TYPE soa_buf[5];

    vstring_sprintf(buf, "%s. %u IN %s ",
		    rr->rname, rr->ttl, dns_strtype(rr->type));
    switch (rr->type) {
    case T_A:
#ifdef T_AAAA
    case T_AAAA:
#endif
	if (dns_rr_to_pa(rr, &host) == 0)
	    msg_fatal("%s: conversion error for resource record type %s: %m",
		      myname, dns_strtype(rr->type));
	vstring_sprintf_append(buf, "%s", host.buf);
	break;
    case T_CNAME:
    case T_DNAME:
    case T_MB:
    case T_MG:
    case T_MR:
    case T_NS:
    case T_PTR:
	vstring_sprintf_append(buf, "%s.", rr->data);
	break;
    case T_TXT:
	vstring_sprintf_append(buf, "%s", rr->data);
	break;
    case T_MX:
	vstring_sprintf_append(buf, "%u %s.", rr->pref, rr->data);
	break;
    case T_TLSA:
	if (rr->data_len >= 3) {
	    uint8_t *ip = (uint8_t *) rr->data;
	    uint8_t usage = *ip++;
	    uint8_t selector = *ip++;
	    uint8_t mtype = *ip++;
	    unsigned i;

	    /* /\.example\. \d+ IN TLSA \d+ \d+ \d+ [\da-f]*$/ IGNORE */
	    vstring_sprintf_append(buf, "%d %d %d ", usage, selector, mtype);
	    for (i = 3; i < rr->data_len; ++i)
		vstring_sprintf_append(buf, "%02x", *ip++);
	} else {
	    vstring_sprintf_append(buf, "[truncated record]");
	}

	/*
	 * We use the SOA record TTL to determine the negative reply TTL. We
	 * save the time fields in the SOA record for debugging, but for now
	 * we don't bother saving the source host and mailbox information, as
	 * that would require changes to the DNS_RR structure. See also code
	 * in dns_get_rr().
	 */
    case T_SOA:
	memcpy(soa_buf, rr->data, sizeof(soa_buf));
	vstring_sprintf_append(buf, "- - %u %u %u %u %u",
			       soa_buf[0], soa_buf[1], soa_buf[2],
			       soa_buf[3], soa_buf[4]);
	break;
    default:
	msg_fatal("%s: don't know how to print type %s",
		  myname, dns_strtype(rr->type));
    }
    return (vstring_str(buf));
}
