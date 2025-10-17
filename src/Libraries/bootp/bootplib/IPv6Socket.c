/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
 * IPv6Socket.c
 * - common functions for creating/sending packets over IPv6 sockets
 */

/* 
 * Modification History
 *
 * May 24, 2013		Dieter Siegmund (dieter@apple.com)
 * - created (based on RTADVSocket.c)
 */

#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/errno.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/sockio.h>
#include <sys/filio.h>
#include <ctype.h>
#include <net/if.h>
#include <net/ethernet.h>
#include <netinet/in.h>
#include <netinet/udp.h>
#include <netinet/in_systm.h>
#include <netinet/ip.h>
#include <net/route.h>
#include <netinet/ip6.h>
#include <netinet6/in6_var.h>
#include <netinet/icmp6.h>
#include <sys/uio.h>
#include "IPConfigurationLog.h"
#include "IPv6Socket.h"
#include "IPv6Sock_Compat.h"
#include "symbol_scope.h"

#define ND_OPT_ALIGN		8
#define BUF_SIZE_NO_HLIM 	CMSG_SPACE(sizeof(struct in6_pktinfo))
#define BUF_SIZE		(BUF_SIZE_NO_HLIM + CMSG_SPACE(sizeof(int)))

PRIVATE_EXTERN int
IPv6SocketSend(int sockfd, int ifindex, const struct sockaddr_in6 * dest,
	       const void * pkt, int pkt_size, int hlim)
{
    struct cmsghdr *	cm;
    char		cmsgbuf[BUF_SIZE];
    struct iovec 	iov;
    struct msghdr 	mhdr;
    ssize_t		n;
    struct in6_pktinfo *pi;
    int			ret;

    /* initialize msghdr for sending packets */
    iov.iov_base = (caddr_t)pkt;
    iov.iov_len = pkt_size;
    mhdr.msg_name = (caddr_t)dest;
    mhdr.msg_namelen = sizeof(struct sockaddr_in6);
    mhdr.msg_flags = 0;
    mhdr.msg_iov = &iov;
    mhdr.msg_iovlen = 1;
    mhdr.msg_control = (caddr_t)cmsgbuf;
    if (hlim >= 0) {
	mhdr.msg_controllen = BUF_SIZE;
    }
    else {
	mhdr.msg_controllen = BUF_SIZE_NO_HLIM;
    }

    /* specify the outgoing interface */
    bzero(cmsgbuf, sizeof(cmsgbuf));
    cm = CMSG_FIRSTHDR(&mhdr);
    if (cm == NULL) {
	/* this can't happen, keep static analyzer happy */
	return (EINVAL);
    }
    cm->cmsg_level = IPPROTO_IPV6;
    cm->cmsg_type = IPV6_PKTINFO;
    cm->cmsg_len = CMSG_LEN(sizeof(struct in6_pktinfo));
    pi = (struct in6_pktinfo *)(void *)CMSG_DATA(cm);
    pi->ipi6_ifindex = ifindex;

    /* specify the hop limit of the packet */
    if (hlim >= 0) {
	cm = CMSG_NXTHDR(&mhdr, cm);
	if (cm == NULL) {
	    /* this can't happen, keep static analyzer happy */
	    return (EINVAL);
	}
	cm->cmsg_level = IPPROTO_IPV6;
	cm->cmsg_type = IPV6_HOPLIMIT;
	cm->cmsg_len = CMSG_LEN(sizeof(int));
	*((int *)(void *)CMSG_DATA(cm)) = hlim;
    }
    n = sendmsg(sockfd, &mhdr, 0);
    if (n != pkt_size) {
	ret = errno;
    }
    else {
	ret = 0;
    }
    return (ret);
}

