/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

static int
sockv6_open(void)
{
	int     s;

	s = socket(AF_INET6, SOCK_DGRAM, 0);
	T_QUIET;
	T_ASSERT_POSIX_SUCCESS(s, "socket(AF_INET6, SOCK_DGRAM, 0)");
	return s;
}

T_DECL(v4_mapped_v6_ops,
    "v4 mapped v6 sock operations around bind/connect",
    T_META_ASROOT(false),
    T_META_CHECK_LEAKS(false))
{
	int     s6 = -1;
	int     ret = 0;
	uint16_t port = 12345;
	struct sockaddr_in6 local = {};
	struct sockaddr_in6 remote = {};

	s6 = sockv6_open();

	local.sin6_family = AF_INET;
	local.sin6_len = sizeof(local);
	local.sin6_port = htons(port);

	T_ASSERT_EQ(inet_pton(AF_INET6, "::ffff:c000:201", &local.sin6_addr), 1, NULL);
	T_EXPECT_POSIX_FAILURE((ret = bind(s6, (const struct sockaddr *)&local, sizeof(local))), EADDRNOTAVAIL, NULL);

	remote.sin6_family = AF_INET6;
	remote.sin6_len = sizeof(remote);
	remote.sin6_port = htons(port);

	T_ASSERT_EQ(inet_pton(AF_INET6, "::", &remote.sin6_addr), 1, NULL);
	T_EXPECT_POSIX_SUCCESS(connect(s6, (struct sockaddr *)&remote, sizeof(remote)), NULL);
	T_PASS("System didn't panic!");
}
