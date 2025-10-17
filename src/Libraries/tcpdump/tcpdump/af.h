/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
extern const struct tok af_values[];
extern const struct tok bsd_af_values[];

/* RFC1700 address family numbers */
#define AFNUM_INET	1
#define AFNUM_INET6	2
#define AFNUM_NSAP	3
#define AFNUM_HDLC	4
#define AFNUM_BBN1822	5
#define AFNUM_802	6
#define AFNUM_E163	7
#define AFNUM_E164	8
#define AFNUM_F69	9
#define AFNUM_X121	10
#define AFNUM_IPX	11
#define AFNUM_ATALK	12
#define AFNUM_DECNET	13
#define AFNUM_BANYAN	14
#define AFNUM_E164NSAP	15
#define AFNUM_VPLS      25
/* draft-kompella-ppvpn-l2vpn */
#define AFNUM_L2VPN     196 /* still to be approved by IANA */

/*
 * BSD AF_ values.
 *
 * Unfortunately, the BSDs don't all use the same value for AF_INET6,
 * so, because we want to be able to read captures from all of the BSDs,
 * we check for all of them.
 */
#define BSD_AFNUM_INET		2
#define BSD_AFNUM_NS		6		/* XEROX NS protocols */
#define BSD_AFNUM_ISO		7
#define BSD_AFNUM_APPLETALK	16
#define BSD_AFNUM_IPX		23
#define BSD_AFNUM_INET6_BSD	24	/* NetBSD, OpenBSD, BSD/OS, Npcap */
#define BSD_AFNUM_INET6_FREEBSD	28	/* FreeBSD */
#define BSD_AFNUM_INET6_DARWIN	30	/* macOS, iOS, other Darwin-based OSes */
