/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 29, 2024.
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
/* System libraries. */

#include <sys_defs.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>

/* DNS library. */

#include <dns.h>

/* dns_rr_to_pa - resource record to printable address */

const char *dns_rr_to_pa(DNS_RR *rr, MAI_HOSTADDR_STR *hostaddr)
{
    if (rr->type == T_A) {
	return (inet_ntop(AF_INET, rr->data, hostaddr->buf,
			  sizeof(hostaddr->buf)));
#ifdef HAS_IPV6
    } else if (rr->type == T_AAAA) {
	return (inet_ntop(AF_INET6, rr->data, hostaddr->buf,
			  sizeof(hostaddr->buf)));
#endif
    } else {
	errno = EAFNOSUPPORT;
	return (0);
    }
}

 /*
  * Stand-alone test program.
  */
#ifdef TEST
#include <vstream.h>
#include <myaddrinfo.h>

static const char *myname;

static NORETURN usage(void)
{
    msg_fatal("usage: %s dnsaddrtype hostname", myname);
}

int     main(int argc, char **argv)
{
    DNS_RR *rr;
    MAI_HOSTADDR_STR hostaddr;
    VSTRING *why;
    int     type;

    myname = argv[0];
    if (argc < 3)
	usage();
    why = vstring_alloc(1);

    while (*++argv) {
	if (argv[1] == 0)
	    usage();
	if ((type = dns_type(argv[0])) == 0)
	    usage();
	if (dns_lookup(argv[1], type, 0, &rr, (VSTRING *) 0, why) != DNS_OK)
	    msg_fatal("%s: %s", argv[1], vstring_str(why));
	if (dns_rr_to_pa(rr, &hostaddr) == 0)
	    msg_fatal("dns_rr_to_sa: %m");
	vstream_printf("%s -> %s\n", argv[1], hostaddr.buf);
	vstream_fflush(VSTREAM_OUT);
	argv += 1;
	dns_rr_free(rr);
    }
    vstring_free(why);
    return (0);
}

#endif
