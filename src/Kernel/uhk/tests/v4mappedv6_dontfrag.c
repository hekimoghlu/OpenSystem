/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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
#define __APPLE_USE_RFC_3542 1

#include <darwintest.h>

#include <sys/ioctl.h>
#include <sys/sysctl.h>
#include <sys/uio.h>

#include <arpa/inet.h>

#include <netinet/ip.h>

#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "net_test_lib.h"

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.net"),
	T_META_ASROOT(true),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("networking"),
	T_META_CHECK_LEAKS(false));

/* need something greater than default MTU */
#define MAX_BUFFER_SIZE 2048

// 169.254
#define LL_NET                 0xa9fe0000 //0x0a000000
// 169.254.1
#define LL_1_NET               (LL_NET | 0x000100)

static void
get_ipv4_address(u_int unit, u_int addr_index, struct in_addr *ip)
{
	/* up to 255 units, 255 addresses */
	ip->s_addr = htonl(LL_1_NET | (unit << 8) | addr_index);
	return;
}

static void
network_interface_init(network_interface_t netif,
    const char * name, unsigned int unit,
    unsigned int address_index)
{
	network_interface_create(netif, name);
	get_ipv4_address(unit, address_index, &netif->ip);
	ifnet_add_ip_address(netif->if_name, netif->ip,
	    inet_class_c_subnet_mask);
	route_add_inet_scoped_subnet(netif->if_name, netif->if_index,
	    netif->ip, inet_class_c_subnet_mask);
}

static network_interface if_one = {0};
static network_interface if_two = {0};

static void
cleanup(void)
{
	network_interface_destroy(&if_one);
	network_interface_destroy(&if_two);
}

T_DECL(v4mappedv6_dontfrag_sockopt, "Tests setting IPV6_DONTFRAG on an IPv4-mapped IPv6 address")
{
	int sockfd = 0;
	ssize_t n;
	char buf[MAX_BUFFER_SIZE] = {0};
	struct sockaddr_in6 sin6 = {0};

	sin6.sin6_len = sizeof(struct sockaddr_in6);
	sin6.sin6_family = AF_INET6;
	sin6.sin6_port = htons(12345);

	T_ATEND(cleanup);

	network_interface_init(&if_one, FETH_NAME, 0, 1);
	network_interface_init(&if_two, FETH_NAME, 0, 2);
	fake_set_peer(if_one.if_name, if_two.if_name);
	ifnet_set_mtu(if_one.if_name, 1500);

	T_ASSERT_EQ(inet_pton(AF_INET6, "::ffff:169.254.1.2", &sin6.sin6_addr), 1, "inet_pton");

	T_ASSERT_POSIX_SUCCESS(sockfd = socket(AF_INET6, SOCK_DGRAM, IPPROTO_UDP), "create socket");

	// This send should succeed (and fragment)
	n = sendto(sockfd, buf, MAX_BUFFER_SIZE, 0,
	    (struct sockaddr *)&sin6, sizeof(sin6));
	T_EXPECT_EQ(n, (ssize_t)MAX_BUFFER_SIZE, "Ensure we wrote MAX_BUFFER_SIZE");

	int Option = 1;
	T_ASSERT_POSIX_SUCCESS(setsockopt(sockfd, IPPROTO_IPV6, IPV6_DONTFRAG, &Option, sizeof(Option)), "setsockopt IPV6_DONTFRAG to 1");

	// This send should fail because MAX_BUFFER_SIZE > MTU and we enabled DONTFRAG
	n = sendto(sockfd, buf, MAX_BUFFER_SIZE, 0,
	    (struct sockaddr *)&sin6, sizeof(sin6));
	T_EXPECT_EQ(errno, EMSGSIZE, "errno should be EMSGSIZE");
	T_EXPECT_EQ(n, (ssize_t)-1, "Expect n of a certain size");

	Option = 0;
	T_ASSERT_POSIX_SUCCESS(setsockopt(sockfd, IPPROTO_IPV6, IPV6_DONTFRAG, &Option, sizeof(Option)), "setsockopt IPV6_DONTFRAG back to 0");

	// This send should succeeed (and fragment) because we turned the option back off
	n = sendto(sockfd, buf, MAX_BUFFER_SIZE, 0,
	    (struct sockaddr *)&sin6, sizeof(sin6));
	T_EXPECT_EQ(n, (ssize_t)MAX_BUFFER_SIZE, "Ensure we wrote MAX_BUFFER_SIZE");
}
