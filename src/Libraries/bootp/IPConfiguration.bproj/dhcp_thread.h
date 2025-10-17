/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
 * dhcp_thread.h
 * - functions implemented in dhcp.c
 * - common type definitions for DHCP-related functions
 */

#ifndef _S_DHCP_THREAD_H
#define _S_DHCP_THREAD_H

/* 
 * Modification History
 *
 * June 26, 2009		Dieter Siegmund (dieter@apple.com)
 * - split out from ipconfigd_threads.h
 */

#include "dhcp_options.h"
#include "ipconfigd_globals.h"
#include "IPv4ClasslessRoute.h"

struct saved_pkt {
    dhcpol_t			options;
    /* ALIGN: align to uint32_t */
    uint32_t			pkt[1500/sizeof(uint32_t)];
    int				pkt_size;
    unsigned 			rating;
    struct in_addr		our_ip;
    struct in_addr		server_ip;
};

void
dhcp_set_default_parameters(uint8_t * params, int n_params);

void
dhcp_set_additional_parameters(uint8_t * params, int n_params);

bool
dhcp_parameter_is_ok(uint8_t param);

void
dhcp_get_lease_from_options(dhcpol_t * options, dhcp_lease_time_t * lease, 
			    dhcp_lease_time_t * t1, dhcp_lease_time_t * t2);

boolean_t
dhcp_get_router_address(dhcpol_t * options_p, struct in_addr our_ip,
			struct in_addr * ret_router_p);

IPv4ClasslessRouteRef
dhcp_copy_classless_routes(dhcpol_t * options_p, int * routes_count_p);

#endif /* _S_DHCP_THREAD_H */
