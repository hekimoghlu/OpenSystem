/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
#include <skywalk/os_skywalk_private.h>
#include <skywalk/nexus/flowswitch/nx_flowswitch.h>
#include <skywalk/nexus/flowswitch/fsw_var.h>

sa_family_t fsw_ip_demux(struct nx_flowswitch *, struct __kern_packet *);

int
fsw_ip_setup(struct nx_flowswitch *fsw, struct ifnet *ifp)
{
#pragma unused(ifp)
	fsw->fsw_resolve = fsw_generic_resolve;
	fsw->fsw_demux = fsw_ip_demux;
	fsw->fsw_frame = NULL;
	fsw->fsw_frame_headroom = 0;
	return 0;
}

sa_family_t
fsw_ip_demux(struct nx_flowswitch *fsw, struct __kern_packet *pkt)
{
#pragma unused(fsw)
	const struct ip *iph;
	const struct ip6_hdr *ip6h;
	sa_family_t af = AF_UNSPEC;
	uint32_t bdlen, bdlim, bdoff;
	uint8_t *baddr;

	MD_BUFLET_ADDR_ABS_DLEN(pkt, baddr, bdlen, bdlim, bdoff);
	baddr += pkt->pkt_headroom;
	iph = (struct ip *)(void *)baddr;
	ip6h = (struct ip6_hdr *)(void *)baddr;

	if ((pkt->pkt_length >= sizeof(*iph)) &&
	    (pkt->pkt_headroom + sizeof(*iph)) <= bdlim &&
	    (iph->ip_v == IPVERSION)) {
		af = AF_INET;
	} else if ((pkt->pkt_length >= sizeof(*ip6h)) &&
	    (pkt->pkt_headroom + sizeof(*ip6h) <= bdlim) &&
	    ((ip6h->ip6_vfc & IPV6_VERSION_MASK) == IPV6_VERSION)) {
		af = AF_INET6;
	} else {
		SK_ERR("unrecognized pkt, hr %u len %u", pkt->pkt_headroom,
		    pkt->pkt_length);
	}

	pkt->pkt_l2_len = 0;

	return af;
}
