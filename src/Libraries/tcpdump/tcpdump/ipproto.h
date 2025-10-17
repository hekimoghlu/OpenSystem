/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
extern const struct tok ipproto_values[];
extern const char *netdb_protoname (const uint8_t);

#ifndef IPPROTO_IP
#define	IPPROTO_IP		0		/* dummy for IP */
#endif
#ifndef IPPROTO_HOPOPTS
#define IPPROTO_HOPOPTS		0		/* IPv6 hop-by-hop options */
#endif
#ifndef IPPROTO_ICMP
#define	IPPROTO_ICMP		1		/* control message protocol */
#endif
#ifndef IPPROTO_IGMP
#define	IPPROTO_IGMP		2		/* group mgmt protocol */
#endif
#ifndef IPPROTO_IPV4
#define IPPROTO_IPV4		4
#endif
#ifndef IPPROTO_TCP
#define	IPPROTO_TCP		6		/* tcp */
#endif
#ifndef IPPROTO_EGP
#define	IPPROTO_EGP		8		/* exterior gateway protocol */
#endif
#ifndef IPPROTO_PIGP
#define IPPROTO_PIGP		9
#endif
#ifndef IPPROTO_UDP
#define	IPPROTO_UDP		17		/* user datagram protocol */
#endif
#ifndef IPPROTO_DCCP
#define	IPPROTO_DCCP		33		/* datagram congestion control protocol */
#endif
#ifndef IPPROTO_IPV6
#define IPPROTO_IPV6		41
#endif
#ifndef IPPROTO_ROUTING
#define IPPROTO_ROUTING		43		/* IPv6 routing header */
#endif
#ifndef IPPROTO_FRAGMENT
#define IPPROTO_FRAGMENT	44		/* IPv6 fragmentation header */
#endif
#ifndef IPPROTO_RSVP
#define IPPROTO_RSVP		46 		/* resource reservation */
#endif
#ifndef IPPROTO_GRE
#define	IPPROTO_GRE		47		/* General Routing Encap. */
#endif
#ifndef IPPROTO_ESP
#define	IPPROTO_ESP		50		/* SIPP Encap Sec. Payload */
#endif
#ifndef IPPROTO_AH
#define	IPPROTO_AH		51		/* SIPP Auth Header */
#endif
#ifndef IPPROTO_MOBILE
#define IPPROTO_MOBILE		55
#endif
#ifndef IPPROTO_ICMPV6
#define IPPROTO_ICMPV6		58		/* ICMPv6 */
#endif
#ifndef IPPROTO_NONE
#define IPPROTO_NONE		59		/* IPv6 no next header */
#endif
#ifndef IPPROTO_DSTOPTS
#define IPPROTO_DSTOPTS		60		/* IPv6 destination options */
#endif
#ifndef IPPROTO_MOBILITY_OLD
/*
 * The current Protocol Numbers list says that the IP protocol number for
 * mobility headers is 135; it cites RFC 6275 (obsoletes RFC 3775).
 *
 * It appears that 62 used to be used, even though that's assigned to
 * a protocol called CFTP; however, the only reference for CFTP is a
 * Network Message from BBN back in 1982, so, for now, we support 62,
 * as well as 135, as a protocol number for mobility headers.
 */
#define IPPROTO_MOBILITY_OLD	62
#endif
#ifndef IPPROTO_ND
#define	IPPROTO_ND		77		/* Sun net disk proto (temp.) */
#endif
#ifndef IPPROTO_EIGRP
#define	IPPROTO_EIGRP		88		/* Cisco/GXS IGRP */
#endif
#ifndef IPPROTO_OSPF
#define IPPROTO_OSPF		89
#endif
#ifndef IPPROTO_PIM
#define IPPROTO_PIM		103
#endif
#ifndef IPPROTO_IPCOMP
#define IPPROTO_IPCOMP		108
#endif
#ifndef IPPROTO_VRRP
#define IPPROTO_VRRP		112 /* See also CARP. */
#endif
#ifndef IPPROTO_PGM
#define IPPROTO_PGM             113
#endif
#ifndef IPPROTO_SCTP
#define IPPROTO_SCTP		132
#endif
#ifndef IPPROTO_MOBILITY
#define IPPROTO_MOBILITY	135
#endif
#ifndef IPPROTO_ETHERNET
#define IPPROTO_ETHERNET	143 /* TEMPORARY - registered 2020-01-31, expires 2021-01-31 */
#endif
