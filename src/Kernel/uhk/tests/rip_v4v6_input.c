/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#include <sys/errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>

#include <darwintest.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.net"),
	T_META_ASROOT(true),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("networking"),
	T_META_CHECK_LEAKS(false),
	T_META_TAG_VM_PREFERRED);


static void
udp_port_scan(void)
{
	int v4_udp_fd;

	T_ASSERT_POSIX_SUCCESS(v4_udp_fd = socket(AF_INET, SOCK_DGRAM, 0),
	    "fd %d = socket(AF_INET, SOCK_DGRAM)", v4_udp_fd);

	char *buffer = "hello";
	size_t len = strlen(buffer) + 1;

	for (in_port_t port = 1; port > 0 && port <= IPPORT_HILASTAUTO; port++) {
		struct sockaddr_in sin = {};
		sin.sin_len = sizeof(struct sockaddr_in);
		sin.sin_family = AF_INET;
		sin.sin_port = htons(port);
		sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

		/*
		 * It is fine to get `ECONNREFUSED` because the port scanning does
		 * trigger ICMP port unreachable messages
		 */
		ssize_t sent = sendto(v4_udp_fd, buffer, len, 0, (struct sockaddr *)&sin, sin.sin_len);
		int saved_errno = errno;
		T_QUIET; T_ASSERT_TRUE(sent >= 0 || errno == ECONNREFUSED, "sendto() to port %u: errno = %d (%s)",
		    port, saved_errno, strerror(saved_errno));
	}

	close(v4_udp_fd);

	T_LOG("udp_port_scan done");
}

static int
open_raw_ipv4_socket(void)
{
	int fd;

	T_ASSERT_POSIX_SUCCESS(fd = socket(AF_INET, SOCK_RAW, 0),
	    "fd %d = socket(AF_INET, SOCK_RAW)", fd);

	return fd;
}

static int
open_raw_ipv6_socket(void)
{
	int fd;

	T_ASSERT_POSIX_SUCCESS(fd = socket(AF_INET6, SOCK_RAW, 0),
	    "fd %d = socket(AF_INET6, SOCK_RAW)", fd);

	int off = 0;
	T_ASSERT_POSIX_SUCCESS(setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(int)),
	    "setsockopt(%d, IPPROTO_IPV6, IPV6_V6ONLY)", fd);

	return fd;
}

static void
close_raw_socket(int fd)
{
	int optval;
	socklen_t optlen = sizeof(optval);

	T_ASSERT_POSIX_SUCCESS(getsockopt(fd, SOL_SOCKET, SO_NUMRCVPKT, &optval, &optlen),
	    "getsockopt(%d, SOL_SOCKET, SO_NUMRCVPKT)", fd);

	T_LOG("fd %d SO_NUMRCVPKT %d", fd, optval);

	(void)close(fd);
}

T_DECL(rip_no_input, "test reception of IPv4 packet on raw IPv6 socket ", T_META_TAG_VM_PREFERRED)
{
	udp_port_scan();

	T_PASS("%s", __func__);
}

T_DECL(rip_v4_input, "test reception of IPv4 packet on raw IPv6 socket ", T_META_TAG_VM_PREFERRED)
{
	int v4_raw_fd1 = open_raw_ipv4_socket();

	udp_port_scan();

	close_raw_socket(v4_raw_fd1);

	T_PASS("%s", __func__);
}

T_DECL(rip_v6_input, "test reception of IPv4 packet on raw IPv6 socket ", T_META_TAG_VM_PREFERRED)
{
	int v6_raw_fd1 = open_raw_ipv6_socket();

	udp_port_scan();

	close_raw_socket(v6_raw_fd1);

	T_PASS("%s", __func__);
}

T_DECL(rip_v4v4_input, "test reception of IPv4 packet on raw IPv6 socket ", T_META_TAG_VM_PREFERRED)
{
	int v4_raw_fd1 = open_raw_ipv4_socket();
	int v4_raw_fd2 = open_raw_ipv4_socket();

	udp_port_scan();

	close_raw_socket(v4_raw_fd1);
	close_raw_socket(v4_raw_fd2);

	T_PASS("%s", __func__);
}

T_DECL(rip_v6v6_input, "test reception of IPv4 packet on raw IPv6 socket ", T_META_TAG_VM_PREFERRED)
{
	int v6_raw_fd1 = open_raw_ipv6_socket();
	int v6_raw_fd2 = open_raw_ipv6_socket();

	udp_port_scan();

	close_raw_socket(v6_raw_fd1);
	close_raw_socket(v6_raw_fd2);

	T_PASS("%s", __func__);
}

T_DECL(rip_v4v6_input, "test reception of IPv4 packet on raw IPv6 socket ", T_META_TAG_VM_PREFERRED)
{
	int v4_raw_fd1 = open_raw_ipv4_socket();
	int v6_raw_fd1 = open_raw_ipv6_socket();

	udp_port_scan();

	close_raw_socket(v4_raw_fd1);
	close_raw_socket(v6_raw_fd1);

	T_PASS("%s", __func__);
}

T_DECL(rip_v4v4v6_input, "test reception of IPv4 packet on raw IPv6 socket ", T_META_TAG_VM_PREFERRED)
{
	int v4_raw_fd1 = open_raw_ipv4_socket();
	int v4_raw_fd2 = open_raw_ipv4_socket();
	int v6_raw_fd = open_raw_ipv6_socket();

	udp_port_scan();

	close_raw_socket(v4_raw_fd1);
	close_raw_socket(v4_raw_fd2);
	close_raw_socket(v6_raw_fd);

	T_PASS("%s", __func__);
}

T_DECL(rip_v4v6v6_input, "test reception of IPv4 packet on raw IPv6 socket ", T_META_TAG_VM_PREFERRED)
{
	int v4_raw_fd1 = open_raw_ipv4_socket();
	int v6_raw_fd1 = open_raw_ipv6_socket();
	int v6_raw_fd2 = open_raw_ipv6_socket();

	udp_port_scan();

	close_raw_socket(v4_raw_fd1);
	close_raw_socket(v6_raw_fd1);
	close_raw_socket(v6_raw_fd2);

	T_PASS("%s", __func__);
}

T_DECL(rip_v4v4v6v6_input, "test reception of IPv4 packet on raw IPv6 socket ", T_META_TAG_VM_PREFERRED)
{
	int v4_raw_fd1 = open_raw_ipv4_socket();
	int v4_raw_fd2 = open_raw_ipv4_socket();
	int v6_raw_fd1 = open_raw_ipv6_socket();
	int v6_raw_fd2 = open_raw_ipv6_socket();

	udp_port_scan();

	close_raw_socket(v4_raw_fd1);
	close_raw_socket(v4_raw_fd2);
	close_raw_socket(v6_raw_fd1);
	close_raw_socket(v6_raw_fd2);

	T_PASS("%s", __func__);
}
