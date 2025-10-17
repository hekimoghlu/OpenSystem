/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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
#include <errno.h>

/* Utility library. */

#include <msg.h>

/* DNS library. */

#include <dns.h>

/* dns_sa_to_rr - socket address to resource record */

DNS_RR *dns_sa_to_rr(const char *hostname, unsigned pref, struct sockaddr *sa)
{
#define DUMMY_TTL	0

    if (sa->sa_family == AF_INET) {
	return (dns_rr_create(hostname, hostname, T_A, C_IN, DUMMY_TTL, pref,
			      (char *) &SOCK_ADDR_IN_ADDR(sa),
			      sizeof(SOCK_ADDR_IN_ADDR(sa))));
#ifdef HAS_IPV6
    } else if (sa->sa_family == AF_INET6) {
	return (dns_rr_create(hostname, hostname, T_AAAA, C_IN, DUMMY_TTL, pref,
			      (char *) &SOCK_ADDR_IN6_ADDR(sa),
			      sizeof(SOCK_ADDR_IN6_ADDR(sa))));
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
#include <stdlib.h>
#include <vstream.h>
#include <myaddrinfo.h>
#include <inet_proto.h>
#include <mymalloc.h>

static const char *myname;

static NORETURN usage(void)
{
    msg_fatal("usage: %s hostname", myname);
}

static int compare_family(const void *a, const void *b)
{
    struct addrinfo *resa = *(struct addrinfo **) a;
    struct addrinfo *resb = *(struct addrinfo **) b;

    return (resa->ai_family - resb->ai_family);
}

int     main(int argc, char **argv)
{
    MAI_HOSTADDR_STR hostaddr;
    struct addrinfo *res0;
    struct addrinfo *res;
    struct addrinfo **resv;
    size_t  len, n;
    DNS_RR *rr;
    int     aierr;

    myname = argv[0];
    if (argc < 2)
	usage();

    inet_proto_init(argv[0], INET_PROTO_NAME_ALL);

    while (*++argv) {
	if ((aierr = hostname_to_sockaddr(argv[0], (char *) 0, 0, &res0)) != 0)
	    msg_fatal("%s: %s", argv[0], MAI_STRERROR(aierr));
	for (len = 0, res = res0; res != 0; res = res->ai_next)
	    len += 1;
	resv = (struct addrinfo **) mymalloc(len * sizeof(*resv));
	for (len = 0, res = res0; res != 0; res = res->ai_next)
	    resv[len++] = res;
	qsort((void *) resv, len, sizeof(*resv), compare_family);
	for (n = 0; n < len; n++) {
	    if ((rr = dns_sa_to_rr(argv[0], 0, resv[n]->ai_addr)) == 0)
		msg_fatal("dns_sa_to_rr: %m");
	    if (dns_rr_to_pa(rr, &hostaddr) == 0)
		msg_fatal("dns_rr_to_pa: %m");
	    vstream_printf("%s -> %s\n", argv[0], hostaddr.buf);
	    vstream_fflush(VSTREAM_OUT);
	    dns_rr_free(rr);
	}
	freeaddrinfo(res0);
	myfree((void *) resv);
    }
    return (0);
}

#endif
