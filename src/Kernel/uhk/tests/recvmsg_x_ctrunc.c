/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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
/* -*- compile-command: "xcrun --sdk macosx.internal make -C tests recvmsg_x_test" -*- */


#define __APPLE_USE_RFC_3542 1

#include <sys/errno.h>
#include <sys/fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sysexits.h>
#include <unistd.h>

#include <arpa/inet.h>

#include <darwintest.h>
#include <darwintest_utils.h>

#define NMSGS       10
#define BUFFERLEN   1000

T_GLOBAL_META(T_META_NAMESPACE("xnu.net"));

static void
send_packets(int sendSocket, u_int packetCount, struct sockaddr *to, int proto)
{
	u_int nmsgs = NMSGS;
	ssize_t sentMsgsCount = 0;
	struct msghdr_x msgList[NMSGS];

	struct msghdr_x *msg;
	struct iovec iovarray[NMSGS];
	char bytes[nmsgs][100];
	const socklen_t cmsg_size = (socklen_t)CMSG_SPACE(sizeof(int));
	char cmsgbuf[NMSGS][cmsg_size];

	bzero(msgList, sizeof(msgList));
	bzero(cmsgbuf, sizeof(cmsgbuf));

	for (int i = 0; i < NMSGS; i++) {
		msg = &msgList[i];

		int dscp = (i % 64) << 2;
		struct cmsghdr *cm;

		cm = (struct cmsghdr *)(void *)&cmsgbuf[i][0];
		if (proto == IPPROTO_IP) {
			cm->cmsg_len = CMSG_LEN(sizeof(int));
			cm->cmsg_level = IPPROTO_IP;
			cm->cmsg_type = IP_TOS;
			*(int *)(void *)CMSG_DATA(cm) = dscp;
			msg->msg_control = cmsgbuf[i];
			msg->msg_controllen = CMSG_SPACE(sizeof(int));
		} else if (proto == IPPROTO_IPV6) {
			cm->cmsg_len = CMSG_LEN(sizeof(sizeof(int)));
			cm->cmsg_level = IPPROTO_IPV6;
			cm->cmsg_type = IPV6_TCLASS;
			*(int *)(void *)CMSG_DATA(cm) = dscp;
			msg->msg_control = cmsgbuf[i];
			msg->msg_controllen = CMSG_SPACE(sizeof(int));
		}

		msg->msg_name = (void *)to;
		msg->msg_namelen = to->sa_len;
		msg->msg_iov = &iovarray[i];
		msg->msg_iovlen = 1;
		iovarray[i].iov_base = &bytes[i];
		iovarray[i].iov_len = 100;
		msg->msg_flags = 0;
	}

	while (1) {
		if (packetCount < nmsgs) {
			nmsgs = packetCount;
		}
		T_EXPECT_POSIX_SUCCESS(sentMsgsCount = sendmsg_x(sendSocket, msgList, nmsgs, 0), "sendmsg_x()");
		if (sentMsgsCount < 0) {
			break;
		} else {
			packetCount -= sentMsgsCount;
		}
		if (packetCount == 0) {
			break;
		}
	}
}

static bool
receive_packets(int recvSocket)
{
	uint8_t maxPacketsToRead = NMSGS;
	int i;
	struct msghdr_x msglist[NMSGS];
	char bytes[NMSGS][100];
	struct iovec vec[NMSGS];
	const socklen_t cmsg_size = (socklen_t)MAX(CMSG_SPACE(sizeof(struct in_pktinfo)), CMSG_SPACE(sizeof(struct in6_pktinfo))) + CMSG_SPACE(sizeof(int));
	char cmsgbuf[NMSGS][cmsg_size];
	struct sockaddr_storage remoteAddress[NMSGS];
	bool success = true;

	bzero(msglist, sizeof(msglist));
	bzero(vec, sizeof(vec));
	bzero(cmsgbuf, sizeof(cmsgbuf));


	ssize_t total_received = 0;
	while (1) {
		ssize_t npkts;

		for (i = 0; i < maxPacketsToRead; i++) {
			struct msghdr_x *msg = &msglist[i];
			vec[i].iov_base = &bytes[i];
			vec[i].iov_len = 100;

			msg->msg_name = &remoteAddress[i];
			msg->msg_namelen = sizeof(struct sockaddr_storage);
			msg->msg_iov = &vec[i];
			msg->msg_iovlen = 1;
			msg->msg_control = cmsgbuf[i];
			msg->msg_controllen = cmsg_size;
		}

		npkts = recvmsg_x(recvSocket, msglist, maxPacketsToRead, 0);
		if (npkts < 0) {
			if (errno == EINTR || errno == EWOULDBLOCK) {
				continue;
			}
			T_EXPECT_POSIX_SUCCESS(npkts, "recvmsg_x() npkts %ld total_received %ld", npkts, total_received);
			break;
		}
		total_received += npkts;

		for (i = 0; i < npkts; i++) {
			struct msghdr_x *msg = &msglist[i];

			if ((msg->msg_controllen < (socklen_t)sizeof(struct cmsghdr)) || (msg->msg_flags & MSG_CTRUNC)) {
				success = false;
				T_LOG("msg[%d] bad  control message len=%d (< %u?) msg_flags 0x%x socket %d",
				    i, msg->msg_controllen, cmsg_size, msg->msg_flags, recvSocket);
			} else {
				T_LOG("msg[%d] good control message len=%d (< %u?) msg_flags 0x%x socket %d",
				    i, msg->msg_controllen, cmsg_size, msg->msg_flags, recvSocket);
			}
			for (struct cmsghdr *cm = (struct cmsghdr *)CMSG_FIRSTHDR(msg);
			    cm != NULL;
			    cm = (struct cmsghdr *)CMSG_NXTHDR(msg, cm)) {
				T_LOG(" cmsg_level %u cmsg_type %u cmsg_len %u",
				    cm->cmsg_level, cm->cmsg_type, cm->cmsg_len);

				if (cm->cmsg_level == IPPROTO_IP &&
				    cm->cmsg_type == IP_RECVTOS &&
				    cm->cmsg_len == CMSG_LEN(sizeof(u_char))) {
					u_char ip_tos = *(u_char *)(void *)CMSG_DATA(cm);

					T_LOG("   ip_tos 0x%x", ip_tos);
				} else if (cm->cmsg_level == IPPROTO_IPV6 &&
				    cm->cmsg_type == IPV6_TCLASS &&
				    cm->cmsg_len == CMSG_LEN(sizeof(int))) {
					int ipv6_tclass = *(int *)(void *)CMSG_DATA(cm);

					T_LOG("   ipv6_tclass 0x%x", ipv6_tclass);
				} else if (cm->cmsg_level == IPPROTO_IPV6 &&
				    cm->cmsg_type == IPV6_PKTINFO &&
				    cm->cmsg_len == CMSG_LEN(sizeof(struct in6_pktinfo))) {
					struct in6_pktinfo *pktinfo = (struct in6_pktinfo *)(void *)CMSG_DATA(cm);
					char addr[40];

					T_LOG("   pktinfo addr %s ifindex %u",
					    inet_ntop(AF_INET6, &pktinfo->ipi6_addr, addr, sizeof(addr)), pktinfo->ipi6_ifindex);
				} else if (cm->cmsg_level == IPPROTO_IP &&
				    cm->cmsg_type == IP_PKTINFO &&
				    cm->cmsg_len == CMSG_LEN(sizeof(struct in_pktinfo))) {
					struct in_pktinfo *pktinfo = (struct in_pktinfo *)(void *)CMSG_DATA(cm);
					char spec_dst[20];
					char addr[20];

					inet_ntop(AF_INET, &pktinfo->ipi_spec_dst, spec_dst, sizeof(spec_dst));
					inet_ntop(AF_INET, &pktinfo->ipi_addr, addr, sizeof(addr));

					T_LOG("   pktinfo ifindex %u spec_dest %s addr %s",
					    pktinfo->ipi_ifindex, spec_dst, addr);
				}
			}
		}

		if (total_received >= maxPacketsToRead) {
			// Since we received max number of packets in the last loop, it is not clear if there
			// are any more left in the socket buffer. So we need to try again
			break;
		}
	}
	return success;
}

T_DECL(recvmsg_x_ipv4_udp, "revcmsg_x() ipv4", T_META_TAG_VM_PREFERRED)
{
	struct sockaddr_in addr = {
		.sin_len = sizeof(addr),
		.sin_family = AF_INET,
		.sin_addr.s_addr = htonl(0x7f000001),
		.sin_port = 0
	};

	int recvSocket;
	T_QUIET; T_EXPECT_POSIX_SUCCESS(recvSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP), "socket()");
	T_QUIET; T_EXPECT_POSIX_SUCCESS(bind(recvSocket, (const struct sockaddr *)&addr, sizeof(addr)), "bind()");

	socklen_t addrLen = sizeof(addr);
	T_QUIET; T_EXPECT_POSIX_SUCCESS(getsockname(recvSocket, (struct sockaddr *)&addr, &addrLen), "getsockname()");

	int one = 1;
	T_QUIET; T_EXPECT_POSIX_SUCCESS(setsockopt(recvSocket, IPPROTO_IP, IP_RECVPKTINFO, (void *)&one, sizeof(one)), "setsockopt(IP_RECVPKTINFO)");

	T_QUIET; T_EXPECT_POSIX_SUCCESS(setsockopt(recvSocket, IPPROTO_IP, IP_RECVTOS, (void *)&one, sizeof(one)), "setsockopt(IP_RECVTOS)");

	int flags = fcntl(recvSocket, F_GETFL, 0);
	T_QUIET; T_EXPECT_POSIX_SUCCESS(fcntl(recvSocket, F_SETFL, flags | O_NONBLOCK), "fcntl()");

	int sendSocket;
	T_QUIET; T_EXPECT_POSIX_SUCCESS(sendSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP), "sendSocket socket()");

	send_packets(sendSocket, 10, (struct sockaddr *)&addr, IPPROTO_IP);

	bool result;
	T_EXPECT_EQ(result = receive_packets(recvSocket), true, "receive_packets");

	close(sendSocket);
	close(recvSocket);
}

T_DECL(recvmsg_x_ipv6_udp, "exercise revcmsg_x() ", T_META_TAG_VM_PREFERRED)
{
	struct sockaddr_in6 addr = {
		.sin6_len = sizeof(addr),
		.sin6_family = AF_INET6,
		.sin6_addr = IN6ADDR_LOOPBACK_INIT,
		.sin6_flowinfo = 0,
		.sin6_scope_id = 0,
		.sin6_port = 0
	};

	int recvSocket;
	T_QUIET; T_EXPECT_POSIX_SUCCESS(recvSocket = socket(AF_INET6, SOCK_DGRAM, IPPROTO_UDP), "socket()");
	T_QUIET; T_EXPECT_POSIX_SUCCESS(bind(recvSocket, (const struct sockaddr *)&addr, sizeof(addr)), "bind()");

	socklen_t addrLen = sizeof(addr);
	T_QUIET; T_EXPECT_POSIX_SUCCESS(getsockname(recvSocket, (struct sockaddr *)&addr, &addrLen), "getsockname()");

	int one = 1;
	T_QUIET; T_EXPECT_POSIX_SUCCESS(setsockopt(recvSocket, IPPROTO_IPV6, IPV6_RECVPKTINFO, (void *)&one, sizeof(one)), "setsockopt(IPV6_RECVPKTINFO)");

	T_QUIET; T_EXPECT_POSIX_SUCCESS(setsockopt(recvSocket, IPPROTO_IPV6, IPV6_RECVTCLASS, (void *)&one, sizeof(one)), "setsockopt(IPV6_RECVTCLASS)");

	int flags = fcntl(recvSocket, F_GETFL, 0);
	T_QUIET; T_EXPECT_POSIX_SUCCESS(fcntl(recvSocket, F_SETFL, flags | O_NONBLOCK), "fcntl()");

	int sendSocket;
	T_QUIET; T_EXPECT_POSIX_SUCCESS(sendSocket = socket(AF_INET6, SOCK_DGRAM, IPPROTO_UDP), "sendSocket socket()");

	send_packets(sendSocket, 10, (struct sockaddr *)&addr, IPPROTO_IPV6);

	bool result;
	T_EXPECT_EQ(result = receive_packets(recvSocket), true, "receive_packets");

	close(sendSocket);
	close(recvSocket);
}
