/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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
#include <darwintest.h>
#include <stdio.h>

#include <darwintest.h>

#include <string.h>
#include <strings.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.net"),
	T_META_ASROOT(true),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("networking"),
	T_META_ENABLED(!TARGET_OS_BRIDGE),
	T_META_CHECK_LEAKS(false));

T_DECL(v4mappedv6_join_group, "Tests setting IPV6_JOIN_GROUP on an IPv4-mapped IPv6 address")
{
	int s;
	struct sockaddr_in6 sin6 = {
		.sin6_family = AF_INET6,
		.sin6_len = sizeof(struct sockaddr_in6),
		.sin6_port = 1337
	};
	struct ipv6_mreq mreq = {};

	T_ASSERT_POSIX_SUCCESS(s = socket(AF_INET6, SOCK_DGRAM, IPPROTO_UDP), "create socket");

	T_ASSERT_POSIX_SUCCESS(inet_pton(AF_INET6, "::ffff:127.0.0.1", &sin6.sin6_addr), "inet_pton");
	T_ASSERT_POSIX_SUCCESS(connect(s, (const struct sockaddr *)&sin6, sizeof(sin6)), "connect");

	memset((unsigned char *)&mreq.ipv6mr_multiaddr, 0xff, 16);

	// This should now fail (but not panic)
	T_ASSERT_POSIX_FAILURE(setsockopt(s, IPPROTO_IPV6, IPV6_JOIN_GROUP, &mreq, sizeof(mreq)), EADDRNOTAVAIL, "setsockopt IPV6_JOIN_GROUP");
}
