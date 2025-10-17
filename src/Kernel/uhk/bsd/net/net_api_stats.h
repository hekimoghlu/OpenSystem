/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
#ifndef __NET_API_STATS__
#define __NET_API_STATS__

#ifdef PRIVATE
#include <stdint.h>

#define NAS_HAS_FLTR_OS_COUNTS 1

/*
 * net_api_stats counts the usage of the networking APIs
 *
 * Note: we are using signed 64 bit values to detect and prevent wrap around
 */
struct net_api_stats {
	/*
	 * Interface Filters
	 */
	int64_t nas_iflt_attach_count;  // Currently attached
	int64_t nas_iflt_attach_os_count;
	int64_t nas_iflt_attach_total;  // Total number of attachments
	int64_t nas_iflt_attach_os_total;

	/*
	 * IP Filters
	 */
	int64_t nas_ipf_add_count;      // Currently attached
	int64_t nas_ipf_add_os_count;
	int64_t nas_ipf_add_total;      // Total number of attachments
	int64_t nas_ipf_add_os_total;

	/*
	 * Socket Filters
	 */
	int64_t nas_sfltr_register_count;       // Currently attached
	int64_t nas_sfltr_register_os_count;
	int64_t nas_sfltr_register_total;       // Total number of attachments
	int64_t nas_sfltr_register_os_total;

	/*
	 * Sockets
	 */
	int64_t nas_socket_alloc_total;
	int64_t nas_socket_in_kernel_total;
	int64_t nas_socket_in_kernel_os_total;
	int64_t nas_socket_necp_clientuuid_total;

	/*
	 * Sockets per protocol domains
	 */
	int64_t nas_socket_domain_local_total;
	int64_t nas_socket_domain_route_total;
	int64_t nas_socket_domain_inet_total;
	int64_t nas_socket_domain_inet6_total;
	int64_t nas_socket_domain_system_total;
	int64_t nas_socket_domain_multipath_total;
	int64_t nas_socket_domain_key_total;
	int64_t nas_socket_domain_ndrv_total;
	int64_t nas_socket_domain_other_total;

	/*
	 * Sockets per domain and type
	 */
	int64_t nas_socket_inet_stream_total;
	int64_t nas_socket_inet_dgram_total;
	int64_t nas_socket_inet_dgram_connected;
	int64_t nas_socket_inet_dgram_dns;      // port 53
	int64_t nas_socket_inet_dgram_no_data;  // typically for interface ioctl

	int64_t nas_socket_inet6_stream_total;
	int64_t nas_socket_inet6_dgram_total;
	int64_t nas_socket_inet6_dgram_connected;
	int64_t nas_socket_inet6_dgram_dns;     // port 53
	int64_t nas_socket_inet6_dgram_no_data; // typically for interface ioctl

	/*
	 * Multicast join
	 */
	int64_t nas_socket_mcast_join_total;
	int64_t nas_socket_mcast_join_os_total;

	/*
	 * IPv6 Extension Header Socket API
	 */
	int64_t nas_sock_inet6_stream_exthdr_in;
	int64_t nas_sock_inet6_stream_exthdr_out;
	int64_t nas_sock_inet6_dgram_exthdr_in;
	int64_t nas_sock_inet6_dgram_exthdr_out;

	/*
	 * Nexus flows
	 */
	int64_t nas_nx_flow_inet_stream_total;
	int64_t nas_nx_flow_inet_dgram_total;

	int64_t nas_nx_flow_inet6_stream_total;
	int64_t nas_nx_flow_inet6_dgram_total;

	/*
	 * Interfaces
	 */
	int64_t nas_ifnet_alloc_count;
	int64_t nas_ifnet_alloc_total;
	int64_t nas_ifnet_alloc_os_count;
	int64_t nas_ifnet_alloc_os_total;

	/*
	 * PF
	 */
	int64_t nas_pf_addrule_total;
	int64_t nas_pf_addrule_os;

	/*
	 * vmnet API
	 */
	int64_t nas_vmnet_total;
};

#ifdef XNU_KERNEL_PRIVATE
extern struct net_api_stats net_api_stats;

/*
 * Increment up to the max value of int64_t
 */
#define INC_ATOMIC_INT64_LIM(counter) {                                 \
	int64_t val;                                                    \
	do {                                                            \
	        val = counter;                                          \
	        if (val >= INT64_MAX) {                                 \
	                break;                                          \
	        }                                                       \
	} while (!OSCompareAndSwap64(val, val + 1, &(counter)));        \
}
#endif /* XNU_KERNEL_PRIVATE */

#endif /* PRIVATE */

#endif /* __NET_API_STATS__ */
