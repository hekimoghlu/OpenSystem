/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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
#ifndef _SKYWALK_NAMESPACE_FLOWIDNS_H_
#define _SKYWALK_NAMESPACE_FLOWIDNS_H_

/*
 * The flowidns (Flow ID namespace) module provides functionality to allocate
 * globally unique identifier for a flow.
 */

typedef uint32_t flowidns_flowid_t;

typedef enum {
	FLOWIDNS_DOMAIN_MIN = 0,
	FLOWIDNS_DOMAIN_IPSEC = FLOWIDNS_DOMAIN_MIN,
	FLOWIDNS_DOMAIN_FLOWSWITCH,
	FLOWIDNS_DOMAIN_INPCB,
	FLOWIDNS_DOMAIN_PF,
	FLOWIDNS_DOMAIN_MAX = FLOWIDNS_DOMAIN_PF
} flowidns_domain_id_t;

struct flowidns_flow_key {
	union {
		struct in_addr  _v4;
		struct in6_addr _v6;
	} ffk_laddr; /* local IP address */
	union {
		struct in_addr  _v4;
		struct in6_addr _v6;
	} ffk_raddr; /* remote IP address */
	union {
		struct {
			uint16_t _lport; /* local port */
			uint16_t _rport; /* remote port */
		} ffk_ports;
		uint32_t ffk_spi; /* IPSec ESP/AH SPI */
		uint32_t ffk_protoid; /* opaque protocol id */
	};
	uint8_t ffk_af; /* IP address family AF_INET* */
	uint8_t ffk_proto; /* IP protocol IP_PROTO_* */
};

#define ffk_laddr_v4    ffk_laddr._v4
#define ffk_laddr_v6    ffk_laddr._v6
#define ffk_raddr_v4    ffk_raddr._v4
#define ffk_raddr_v6    ffk_raddr._v6
#define ffk_lport       ffk_ports._lport
#define ffk_rport       ffk_ports._rport

extern int flowidns_init(void);
extern void flowidns_fini(void);

/*
 * Allocate a globally unique flow identifier.
 */
extern void flowidns_allocate_flowid(flowidns_domain_id_t domain,
    struct flowidns_flow_key *flow_key, flowidns_flowid_t *flowid);

/*
 * Release an allocated flow identifier.
 */
extern void flowidns_release_flowid(flowidns_flowid_t flowid);

#endif /* !_SKYWALK_NAMESPACE_FLOWIDNS_H_ */
