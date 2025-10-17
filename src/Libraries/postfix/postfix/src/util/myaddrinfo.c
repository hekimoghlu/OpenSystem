/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 4, 2022.
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
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>			/* sprintf() */

/* Utility library. */

#include <mymalloc.h>
#include <valid_hostname.h>
#include <sock_addr.h>
#include <stringops.h>
#include <msg.h>
#include <inet_proto.h>
#include <myaddrinfo.h>
#include <split_at.h>

/* Application-specific. */

 /*
  * Use an old trick to save some space: allocate space for two objects in
  * one. In Postfix we often use this trick for structures that have an array
  * of things at the end.
  */
struct ipv4addrinfo {
    struct addrinfo info;
    struct sockaddr_in sin;
};

 /*
  * When we're not interested in service ports, we must pick a socket type
  * otherwise getaddrinfo() will give us duplicate results: one set for TCP,
  * and another set for UDP. For consistency, we'll use the same default
  * socket type for the results from emulation mode.
  */
#define MAI_SOCKTYPE	SOCK_STREAM	/* getaddrinfo() query */

#ifdef EMULATE_IPV4_ADDRINFO

/* clone_ipv4addrinfo - clone ipv4addrinfo structure */

static struct ipv4addrinfo *clone_ipv4addrinfo(struct ipv4addrinfo * tp)
{
    struct ipv4addrinfo *ip;

    ip = (struct ipv4addrinfo *) mymalloc(sizeof(*ip));
    *ip = *tp;
    ip->info.ai_addr = (struct sockaddr *) &(ip->sin);
    return (ip);
}

/* init_ipv4addrinfo - initialize an ipv4addrinfo structure */

static void init_ipv4addrinfo(struct ipv4addrinfo * ip, int socktype)
{

    /*
     * Portability: null pointers aren't necessarily all-zero bits, so we
     * make explicit assignments to all the pointers that we're aware of.
     */
    memset((void *) ip, 0, sizeof(*ip));
    ip->info.ai_family = PF_INET;
    ip->info.ai_socktype = socktype;
    ip->info.ai_protocol = 0;			/* XXX */
    ip->info.ai_addrlen = sizeof(ip->sin);
    ip->info.ai_canonname = 0;
    ip->info.ai_addr = (struct sockaddr *) &(ip->sin);
    ip->info.ai_next = 0;
    ip->sin.sin_family = AF_INET;
#ifdef HAS_SA_LEN
    ip->sin.sin_len = sizeof(ip->sin);
#endif
}

/* find_service - translate numeric or symbolic service name */

static int find_service(const char *service, int socktype)
{
    struct servent *sp;
    const char *proto;
    unsigned port;

    if (alldig(service)) {
	port = atoi(service);
	return (port < 65536 ? htons(port) : -1);
    }
    if (socktype == SOCK_STREAM) {
	proto = "tcp";
    } else if (socktype == SOCK_DGRAM) {
	proto = "udp";
    } else {
	return (-1);
    }
    if ((sp = getservbyname(service, proto)) != 0) {
	return (sp->s_port);
    } else {
	return (-1);
    }
}

#endif

/* hostname_to_sockaddr_pf - hostname to binary address form */

int     hostname_to_sockaddr_pf(const char *hostname, int pf,
				        const char *service, int socktype,
				        struct addrinfo ** res)
{
#ifdef EMULATE_IPV4_ADDRINFO

    /*
     * Emulated getaddrinfo(3) version.
     */
    static struct ipv4addrinfo template;
    struct ipv4addrinfo *ip;
    struct ipv4addrinfo *prev;
    struct in_addr addr;
    struct hostent *hp;
    char  **name_list;
    int     port;

    /*
     * Validate the service.
     */
    if (service) {
	if ((port = find_service(service, socktype)) < 0)
	    return (EAI_SERVICE);
    } else {
	port = 0;
	socktype = MAI_SOCKTYPE;
    }

    /*
     * No host means INADDR_ANY.
     */
    if (hostname == 0) {
	ip = (struct ipv4addrinfo *) mymalloc(sizeof(*ip));
	init_ipv4addrinfo(ip, socktype);
	ip->sin.sin_addr.s_addr = INADDR_ANY;
	ip->sin.sin_port = port;
	*res = &(ip->info);
	return (0);
    }

    /*
     * Numeric host.
     */
    if (inet_pton(AF_INET, hostname, (void *) &addr) == 1) {
	ip = (struct ipv4addrinfo *) mymalloc(sizeof(*ip));
	init_ipv4addrinfo(ip, socktype);
	ip->sin.sin_addr = addr;
	ip->sin.sin_port = port;
	*res = &(ip->info);
	return (0);
    }

    /*
     * Look up the IPv4 address list.
     */
    if ((hp = gethostbyname(hostname)) == 0)
	return (h_errno == TRY_AGAIN ? EAI_AGAIN : EAI_NODATA);
    if (hp->h_addrtype != AF_INET
	|| hp->h_length != sizeof(template.sin.sin_addr))
	return (EAI_NODATA);

    /*
     * Initialize the result template.
     */
    if (template.info.ai_addrlen == 0)
	init_ipv4addrinfo(&template, socktype);

    /*
     * Copy the address information into an addrinfo structure.
     */
    prev = &template;
    for (name_list = hp->h_addr_list; name_list[0]; name_list++) {
	ip = clone_ipv4addrinfo(prev);
	ip->sin.sin_addr = IN_ADDR(name_list[0]);
	ip->sin.sin_port = port;
	if (prev == &template)
	    *res = &(ip->info);
	else
	    prev->info.ai_next = &(ip->info);
	prev = ip;
    }
    return (0);
#else

    /*
     * Native getaddrinfo(3) version.
     * 
     * XXX Wild-card listener issues.
     * 
     * With most IPv4 plus IPv6 systems, an IPv6 wild-card listener also listens
     * on the IPv4 wild-card address. Connections from IPv4 clients appear as
     * IPv4-in-IPv6 addresses; when Postfix support for IPv4 is turned on,
     * Postfix automatically maps these embedded addresses to their original
     * IPv4 form. So everything seems to be fine.
     * 
     * However, some applications prefer to use separate listener sockets for
     * IPv4 and IPv6. The Postfix IPv6 patch provided such an example. And
     * this is where things become tricky. On many systems the IPv6 and IPv4
     * wild-card listeners cannot coexist. When one is already active, the
     * other fails with EADDRINUSE. Solaris 9, however, will automagically
     * "do the right thing" and allow both listeners to coexist.
     * 
     * Recent systems have the IPV6_V6ONLY feature (RFC 3493), which tells the
     * system that we really mean IPv6 when we say IPv6. This allows us to
     * set up separate wild-card listener sockets for IPv4 and IPv6. So
     * everything seems to be fine again.
     * 
     * The following workaround disables the wild-card IPv4 listener when
     * IPV6_V6ONLY is unavailable. This is necessary for some Linux versions,
     * but is not needed for Solaris 9 (which allows IPv4 and IPv6 wild-card
     * listeners to coexist). Solaris 10 beta already has IPV6_V6ONLY.
     * 
     * XXX This workaround obviously breaks if we want to support protocols in
     * addition to IPv6 and IPv4, but it is needed only until IPv6
     * implementations catch up with RFC 3493. A nicer fix is to filter the
     * getaddrinfo() result, and to return a vector of addrinfo pointers to
     * only those types of elements that the caller has expressed interested
     * in.
     * 
     * XXX Vanilla AIX 5.1 getaddrinfo() does not support a null hostname with
     * AI_PASSIVE. And since we don't know how getaddrinfo() manages its
     * memory we can't bypass it for this special case, or freeaddrinfo()
     * might blow up. Instead we turn off IPV6_V6ONLY in inet_listen(), and
     * supply a protocol-dependent hard-coded string value to getaddrinfo()
     * below, so that it will convert into the appropriate wild-card address.
     * 
     * XXX AIX 5.[1-3] getaddrinfo() may return a non-null port when a null
     * service argument is specified.
     */
    struct addrinfo hints;
    int     err;

    memset((void *) &hints, 0, sizeof(hints));
    hints.ai_family = (pf != PF_UNSPEC) ? pf : inet_proto_info()->ai_family;
    hints.ai_socktype = service ? socktype : MAI_SOCKTYPE;
    if (!hostname) {
	hints.ai_flags = AI_PASSIVE;
#if !defined(IPV6_V6ONLY) || defined(BROKEN_AI_PASSIVE_NULL_HOST)
	switch (hints.ai_family) {
	case PF_UNSPEC:
	    hints.ai_family = PF_INET6;
#ifdef BROKEN_AI_PASSIVE_NULL_HOST
	case PF_INET6:
	    hostname = "::";
	    break;
	case PF_INET:
	    hostname = "0.0.0.0";
	    break;
#endif
	}
#endif
    }
    err = getaddrinfo(hostname, service, &hints, res);
#if defined(BROKEN_AI_NULL_SERVICE)
    if (service == 0 && err == 0) {
	struct addrinfo *r;
	unsigned short *portp;

	for (r = *res; r != 0; r = r->ai_next)
	    if (*(portp = SOCK_ADDR_PORTP(r->ai_addr)) != 0)
		*portp = 0;
    }
#endif
    return (err);
#endif
}

/* hostaddr_to_sockaddr - printable address to binary address form */

int     hostaddr_to_sockaddr(const char *hostaddr, const char *service,
			             int socktype, struct addrinfo ** res)
{
#ifdef EMULATE_IPV4_ADDRINFO

    /*
     * Emulated getaddrinfo(3) version.
     */
    struct ipv4addrinfo *ip;
    struct in_addr addr;
    int     port;

    /*
     * Validate the service.
     */
    if (service) {
	if ((port = find_service(service, socktype)) < 0)
	    return (EAI_SERVICE);
    } else {
	port = 0;
	socktype = MAI_SOCKTYPE;
    }

    /*
     * No host means INADDR_ANY.
     */
    if (hostaddr == 0) {
	ip = (struct ipv4addrinfo *) mymalloc(sizeof(*ip));
	init_ipv4addrinfo(ip, socktype);
	ip->sin.sin_addr.s_addr = INADDR_ANY;
	ip->sin.sin_port = port;
	*res = &(ip->info);
	return (0);
    }

    /*
     * Deal with bad address forms.
     */
    switch (inet_pton(AF_INET, hostaddr, (void *) &addr)) {
    case 1:					/* Success */
	break;
    default:					/* Unparsable */
	return (EAI_NONAME);
    case -1:					/* See errno */
	return (EAI_SYSTEM);
    }

    /*
     * Initialize the result structure.
     */
    ip = (struct ipv4addrinfo *) mymalloc(sizeof(*ip));
    init_ipv4addrinfo(ip, socktype);

    /*
     * And copy the result.
     */
    ip->sin.sin_addr = addr;
    ip->sin.sin_port = port;
    *res = &(ip->info);

    return (0);
#else

    /*
     * Native getaddrinfo(3) version. See comments in hostname_to_sockaddr().
     * 
     * XXX Vanilla AIX 5.1 getaddrinfo() returns multiple results when
     * converting a printable ipv4 or ipv6 address to socket address with
     * ai_family=PF_UNSPEC, ai_flags=AI_NUMERICHOST, ai_socktype=SOCK_STREAM,
     * ai_protocol=0 or IPPROTO_TCP, and service=0. The workaround is to
     * ignore all but the first result.
     * 
     * XXX AIX 5.[1-3] getaddrinfo() may return a non-null port when a null
     * service argument is specified.
     */
    struct addrinfo hints;
    int     err;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = inet_proto_info()->ai_family;
    hints.ai_socktype = service ? socktype : MAI_SOCKTYPE;
    hints.ai_flags = AI_NUMERICHOST;
    if (!hostaddr) {
	hints.ai_flags |= AI_PASSIVE;
#if !defined(IPV6_V6ONLY) || defined(BROKEN_AI_PASSIVE_NULL_HOST)
	switch (hints.ai_family) {
	case PF_UNSPEC:
	    hints.ai_family = PF_INET6;
#ifdef BROKEN_AI_PASSIVE_NULL_HOST
	case PF_INET6:
	    hostaddr = "::";
	    break;
	case PF_INET:
	    hostaddr = "0.0.0.0";
	    break;
#endif
	}
#endif
    }
    err = getaddrinfo(hostaddr, service, &hints, res);
#if defined(BROKEN_AI_NULL_SERVICE)
    if (service == 0 && err == 0) {
	struct addrinfo *r;
	unsigned short *portp;

	for (r = *res; r != 0; r = r->ai_next)
	    if (*(portp = SOCK_ADDR_PORTP(r->ai_addr)) != 0)
		*portp = 0;
    }
#endif
    return (err);
#endif
}

/* sockaddr_to_hostaddr - binary address to printable address form */

int     sockaddr_to_hostaddr(const struct sockaddr *sa, SOCKADDR_SIZE salen,
			             MAI_HOSTADDR_STR *hostaddr,
			             MAI_SERVPORT_STR *portnum,
			             int unused_socktype)
{
#ifdef EMULATE_IPV4_ADDRINFO
    char    portbuf[sizeof("65535")];
    ssize_t len;

    /*
     * Emulated getnameinfo(3) version. The buffer length includes the space
     * for the null terminator.
     */
    if (sa->sa_family != AF_INET) {
	errno = EAFNOSUPPORT;
	return (EAI_SYSTEM);
    }
    if (hostaddr != 0) {
	if (inet_ntop(AF_INET, (void *) &(SOCK_ADDR_IN_ADDR(sa)),
		      hostaddr->buf, sizeof(hostaddr->buf)) == 0)
	    return (EAI_SYSTEM);
    }
    if (portnum != 0) {
	sprintf(portbuf, "%d", ntohs(SOCK_ADDR_IN_PORT(sa)) & 0xffff);
	if ((len = strlen(portbuf)) >= sizeof(portnum->buf)) {
	    errno = ENOSPC;
	    return (EAI_SYSTEM);
	}
	memcpy(portnum->buf, portbuf, len + 1);
    }
    return (0);
#else
    int     ret;

    /*
     * Native getnameinfo(3) version.
     */
    ret = getnameinfo(sa, salen,
		      hostaddr ? hostaddr->buf : (char *) 0,
		      hostaddr ? sizeof(hostaddr->buf) : 0,
		      portnum ? portnum->buf : (char *) 0,
		      portnum ? sizeof(portnum->buf) : 0,
		      NI_NUMERICHOST | NI_NUMERICSERV);
    if (hostaddr != 0 && ret == 0 && sa->sa_family == AF_INET6)
	(void) split_at(hostaddr->buf, '%');
    return (ret);
#endif
}

/* sockaddr_to_hostname - binary address to printable hostname */

int     sockaddr_to_hostname(const struct sockaddr *sa, SOCKADDR_SIZE salen,
			             MAI_HOSTNAME_STR *hostname,
			             MAI_SERVNAME_STR *service,
			             int socktype)
{
#ifdef EMULATE_IPV4_ADDRINFO

    /*
     * Emulated getnameinfo(3) version.
     */
    struct hostent *hp;
    struct servent *sp;
    size_t  len;

    /*
     * Sanity check.
     */
    if (sa->sa_family != AF_INET)
	return (EAI_NODATA);

    /*
     * Look up the host name.
     */
    if (hostname != 0) {
	if ((hp = gethostbyaddr((char *) &(SOCK_ADDR_IN_ADDR(sa)),
				sizeof(SOCK_ADDR_IN_ADDR(sa)),
				AF_INET)) == 0)
	    return (h_errno == TRY_AGAIN ? EAI_AGAIN : EAI_NONAME);

	/*
	 * Save the result. The buffer length includes the space for the null
	 * terminator. Hostname sanity checks are at the end of this
	 * function.
	 */
	if ((len = strlen(hp->h_name)) >= sizeof(hostname->buf)) {
	    errno = ENOSPC;
	    return (EAI_SYSTEM);
	}
	memcpy(hostname->buf, hp->h_name, len + 1);
    }

    /*
     * Look up the service.
     */
    if (service != 0) {
	if ((sp = getservbyport(ntohs(SOCK_ADDR_IN_PORT(sa)),
			      socktype == SOCK_DGRAM ? "udp" : "tcp")) == 0)
	    return (EAI_NONAME);

	/*
	 * Save the result. The buffer length includes the space for the null
	 * terminator.
	 */
	if ((len = strlen(sp->s_name)) >= sizeof(service->buf)) {
	    errno = ENOSPC;
	    return (EAI_SYSTEM);
	}
	memcpy(service->buf, sp->s_name, len + 1);
    }
#else

    /*
     * Native getnameinfo(3) version.
     */
    int     err;

    err = getnameinfo(sa, salen,
		      hostname ? hostname->buf : (char *) 0,
		      hostname ? sizeof(hostname->buf) : 0,
		      service ? service->buf : (char *) 0,
		      service ? sizeof(service->buf) : 0,
		      socktype == SOCK_DGRAM ?
		      NI_NAMEREQD | NI_DGRAM : NI_NAMEREQD);
    if (err != 0)
	return (err);
#endif

    /*
     * Hostname sanity checks.
     */
    if (hostname != 0) {
	if (valid_hostaddr(hostname->buf, DONT_GRIPE)) {
	    msg_warn("numeric hostname: %s", hostname->buf);
	    return (EAI_NONAME);
	}
	if (!valid_hostname(hostname->buf, DO_GRIPE))
	    return (EAI_NONAME);
    }
    return (0);
}

/* myaddrinfo_control - fine control */

void    myaddrinfo_control(int name,...)
{
    const char *myname = "myaddrinfo_control";
    va_list ap;

    for (va_start(ap, name); name != 0; name = va_arg(ap, int)) {
	switch (name) {
	default:
	    msg_panic("%s: bad name %d", myname, name);
	}
    }
    va_end(ap);
}

#ifdef EMULATE_IPV4_ADDRINFO

/* freeaddrinfo - release storage */

void    freeaddrinfo(struct addrinfo * ai)
{
    struct addrinfo *ap;
    struct addrinfo *next;

    /*
     * Artefact of implementation: tolerate a null pointer argument.
     */
    for (ap = ai; ap != 0; ap = next) {
	next = ap->ai_next;
	if (ap->ai_canonname)
	    myfree(ap->ai_canonname);
	/* ap->ai_addr is allocated within this memory block */
	myfree((void *) ap);
    }
}

static char *ai_errlist[] = {
    "Success",
    "Address family for hostname not supported",	/* EAI_ADDRFAMILY */
    "Temporary failure in name resolution",	/* EAI_AGAIN	 */
    "Invalid value for ai_flags",	/* EAI_BADFLAGS   */
    "Non-recoverable failure in name resolution",	/* EAI_FAIL	 */
    "ai_family not supported",		/* EAI_FAMILY     */
    "Memory allocation failure",	/* EAI_MEMORY     */
    "No address associated with hostname",	/* EAI_NODATA     */
    "hostname nor servname provided, or not known",	/* EAI_NONAME     */
    "service name not supported for ai_socktype",	/* EAI_SERVICE    */
    "ai_socktype not supported",	/* EAI_SOCKTYPE   */
    "System error returned in errno",	/* EAI_SYSTEM     */
    "Invalid value for hints",		/* EAI_BADHINTS   */
    "Resolved protocol is unknown",	/* EAI_PROTOCOL   */
    "Unknown error",			/* EAI_MAX	  */
};

/* gai_strerror - error number to string */

char   *gai_strerror(int ecode)
{

    /*
     * Note: EAI_SYSTEM errors are not automatically handed over to
     * strerror(). The application decides.
     */
    if (ecode < 0 || ecode > EAI_MAX)
	ecode = EAI_MAX;
    return (ai_errlist[ecode]);
}

#endif

#ifdef TEST

 /*
  * A test program that takes some info from the command line and runs it
  * forward and backward through the above conversion routines.
  */
#include <stdlib.h>
#include <msg.h>
#include <vstream.h>
#include <msg_vstream.h>

static int compare_family(const void *a, const void *b)
{
    struct addrinfo *resa = *(struct addrinfo **) a;
    struct addrinfo *resb = *(struct addrinfo **) b;

    return (resa->ai_family - resb->ai_family);
}

int     main(int argc, char **argv)
{
    struct addrinfo *info;
    struct addrinfo *ip;
    struct addrinfo **resv;
    MAI_HOSTNAME_STR host;
    MAI_HOSTADDR_STR addr;
    size_t  len, n;
    int     err;

    msg_vstream_init(argv[0], VSTREAM_ERR);

    if (argc != 4)
	msg_fatal("usage: %s protocols hostname hostaddress", argv[0]);

    inet_proto_init(argv[0], argv[1]);

    msg_info("=== hostname %s ===", argv[2]);

    if ((err = hostname_to_sockaddr(argv[2], (char *) 0, 0, &info)) != 0) {
	msg_info("hostname_to_sockaddr(%s): %s",
	  argv[2], err == EAI_SYSTEM ? strerror(errno) : gai_strerror(err));
    } else {
	for (len = 0, ip = info; ip != 0; ip = ip->ai_next)
	    len += 1;
	resv = (struct addrinfo **) mymalloc(len * sizeof(*resv));
	for (len = 0, ip = info; ip != 0; ip = ip->ai_next)
	    resv[len++] = ip;
	qsort((void *) resv, len, sizeof(*resv), compare_family);
	for (n = 0; n < len; n++) {
	    ip = resv[n];
	    if ((err = sockaddr_to_hostaddr(ip->ai_addr, ip->ai_addrlen, &addr,
					 (MAI_SERVPORT_STR *) 0, 0)) != 0) {
		msg_info("sockaddr_to_hostaddr: %s",
		   err == EAI_SYSTEM ? strerror(errno) : gai_strerror(err));
		continue;
	    }
	    msg_info("%s -> family=%d sock=%d proto=%d %s", argv[2],
		 ip->ai_family, ip->ai_socktype, ip->ai_protocol, addr.buf);
	    if ((err = sockaddr_to_hostname(ip->ai_addr, ip->ai_addrlen, &host,
					 (MAI_SERVNAME_STR *) 0, 0)) != 0) {
		msg_info("sockaddr_to_hostname: %s",
		   err == EAI_SYSTEM ? strerror(errno) : gai_strerror(err));
		continue;
	    }
	    msg_info("%s -> %s", addr.buf, host.buf);
	}
	freeaddrinfo(info);
	myfree((void *) resv);
    }

    msg_info("=== host address %s ===", argv[3]);

    if ((err = hostaddr_to_sockaddr(argv[3], (char *) 0, 0, &ip)) != 0) {
	msg_info("hostaddr_to_sockaddr(%s): %s",
	  argv[3], err == EAI_SYSTEM ? strerror(errno) : gai_strerror(err));
    } else {
	if ((err = sockaddr_to_hostaddr(ip->ai_addr, ip->ai_addrlen, &addr,
					(MAI_SERVPORT_STR *) 0, 0)) != 0) {
	    msg_info("sockaddr_to_hostaddr: %s",
		   err == EAI_SYSTEM ? strerror(errno) : gai_strerror(err));
	} else {
	    msg_info("%s -> family=%d sock=%d proto=%d %s", argv[3],
		 ip->ai_family, ip->ai_socktype, ip->ai_protocol, addr.buf);
	    if ((err = sockaddr_to_hostname(ip->ai_addr, ip->ai_addrlen, &host,
					 (MAI_SERVNAME_STR *) 0, 0)) != 0) {
		msg_info("sockaddr_to_hostname: %s",
		   err == EAI_SYSTEM ? strerror(errno) : gai_strerror(err));
	    } else
		msg_info("%s -> %s", addr.buf, host.buf);
	    freeaddrinfo(ip);
	}
    }
    exit(0);
}

#endif
