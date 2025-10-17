/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
#include <sys/systm.h>
#include <sys/types.h>
#include <sys/random.h>

#include <skywalk/os_skywalk_private.h>

#include <netinet/in.h>
#include <netinet/in_var.h>
#include <netinet/in_pcb.h> /* for ipport_firstauto and ipport_lastauto */

#include <skywalk/nexus/flowswitch/flow/flow_var.h>

/*
 * caller needs to do local addr resolution (e.g. hostos_sk_source_addr_select)
 * before calling this if a specific laddr is to be reserved. Otherwise it
 * would bind to ADDR_ANY.
 */
int
flow_namespace_create(union sockaddr_in_4_6 *laddr, uint8_t protocol,
    netns_token *token, uint16_t nfr_flags, struct ns_flow_info *nfi)
{
	sa_family_t af = laddr->sa.sa_family;
	uint32_t *addr;
	uint32_t netns_rsv_flags = NETNS_SKYWALK;
	uint8_t addr_len;
	int err;
	int so_type = 0;

	*token = NULL;

	if (__improbable(!netns_is_enabled())) {
		SK_ERR("netns is not enabled");
		return ENOTSUP;
	}

	if (nfr_flags & NXFLOWREQF_LISTENER) {
		netns_rsv_flags = NETNS_LISTENER;
	}
	if (nfr_flags & NXFLOWREQF_NOWAKEFROMSLEEP) {
		netns_rsv_flags |= NETNS_NOWAKEFROMSLEEP;
	}
	if (nfr_flags & NXFLOWREQF_REUSEPORT) {
		netns_rsv_flags |= NETNS_REUSEPORT;
	}

	/* validate protocol */
	switch (protocol) {
	case IPPROTO_UDP:
		so_type = SOCK_DGRAM;
		break;

	case IPPROTO_TCP:
		so_type = SOCK_STREAM;
		break;

	default:
		SK_ERR("invalid protocol (%d)", protocol);
		return EINVAL;
	}

	/* set up addresses */
	switch (af) {
	case AF_INET:
		addr = (uint32_t *)&laddr->sin.sin_addr;
		addr_len = 4;
		break;

	case AF_INET6:
		addr = (uint32_t *)&laddr->sin6.sin6_addr;
		addr_len = 16;
		break;

	default:
		SK_ERR("invalid src address family (%d)", laddr->sa.sa_family);
		return EINVAL;
	}

	/* Assign an ephemeral port, if no port was specified */
	if (laddr->sin.sin_port == 0) {
		err = netns_reserve_ephemeral(token, addr, addr_len, protocol,
		    &laddr->sin.sin_port, netns_rsv_flags, nfi);
	} else {
		err = netns_reserve(token, addr, addr_len, protocol,
		    laddr->sin.sin_port, netns_rsv_flags, nfi);
	}

	SK_DF(SK_VERB_FLOW, "token (%s port %d) BIND",
	    (protocol == IPPROTO_TCP) ? "tcp" : "udp",
	    ntohs(laddr->sin.sin_port));

	return err;
}

void
flow_namespace_destroy(netns_token *token)
{
	netns_release(token);
}

void
flow_namespace_half_close(netns_token *token)
{
	if (NETNS_TOKEN_VALID(token)) {
		netns_half_close(token);
	}
}

void
flow_namespace_withdraw(netns_token *token)
{
	if (NETNS_TOKEN_VALID(token)) {
		netns_withdraw(token);
	}
}
