/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include <net/route.h>
#include <sys/socket.h>
#include <unistd.h>

#include <darwintest.h>

#define ROUNDUP32(n) (((n) + sizeof(uint32_t) - 1) & ~(sizeof(uint32_t) - 1))

T_DECL(route_output_stack_oflow_56033075, "Stack overflow via ma_copy through route_output")
{
	int s;
	uint8_t buf[
		sizeof(struct rt_msghdr) +
		ROUNDUP32(sizeof(struct sockaddr_storage) + 1) + /* RTAX_DST */
		ROUNDUP32(sizeof(struct sockaddr_storage) + 1) + /* RTAX_GATEWAY */
		ROUNDUP32(sizeof(struct sockaddr_storage) + 1)   /* RTAX_NETMASK */
	];
	struct rt_msghdr *rtm = (struct rt_msghdr *)buf;
	struct sockaddr *sa;
	size_t len;

	bzero(buf, sizeof(buf));
	rtm->rtm_type = RTM_GET;
	rtm->rtm_version = RTM_VERSION;
	rtm->rtm_addrs = RTA_DST | RTA_GATEWAY | RTA_NETMASK;
	len = sizeof(struct rt_msghdr);

	/* RTAX_DST: */
	sa = (struct sockaddr *)(rtm + 1);
	sa->sa_family = AF_INET6;
	sa->sa_len = sizeof(struct sockaddr_storage) + 1;
	memset(&sa->sa_data[0], 0xff, sa->sa_len);
	len += ROUNDUP32(sa->sa_len);

	/* RTAX_GATEWAY: */
	sa = (struct sockaddr *)((void *)buf + len);
	sa->sa_family = AF_INET6;
	sa->sa_len = sizeof(struct sockaddr_storage) + 1;
	memset(&sa->sa_data[0], 0xff, sa->sa_len);
	len += ROUNDUP32(sa->sa_len);

	/* RTAX_NETMASK: */
	sa = (struct sockaddr *)((void *)buf + len);
	sa->sa_family = AF_INET6;
	sa->sa_len = sizeof(struct sockaddr_storage) + 1;
	memset(&sa->sa_data[0], 0x41, sa->sa_len);
	len += ROUNDUP32(sa->sa_len);

	T_SETUPBEGIN;
	T_ASSERT_POSIX_SUCCESS(s = socket(PF_ROUTE, SOCK_RAW, PF_ROUTE), NULL);
	T_SETUPEND;

	/* check we get EINVAL for > sizeof(struct sockaddr_storage): */
	rtm->rtm_msglen = len;
	T_ASSERT_EQ(-1, send(s, buf, len, 0), NULL);
	T_ASSERT_EQ(EINVAL, errno, NULL);

	/* now check the ok case: */
	len = sizeof(struct rt_msghdr);

	/* RTAX_DST: */
	sa = (struct sockaddr *)(rtm + 1);
	sa->sa_family = AF_INET6;
	sa->sa_len = sizeof(struct sockaddr_storage);
	len += ROUNDUP32(sa->sa_len);

	/* RTAX_GATEWAY: */
	sa = (struct sockaddr *)((void *)buf + len);
	sa->sa_family = AF_INET6;
	sa->sa_len = sizeof(struct sockaddr_storage);
	len += ROUNDUP32(sa->sa_len);

	/* RTAX_NETMASK: */
	sa = (struct sockaddr *)((void *)buf + len);
	sa->sa_family = AF_INET6;
	sa->sa_len = sizeof(struct sockaddr_storage);
	len += ROUNDUP32(sa->sa_len);

	rtm->rtm_msglen = len;
	T_ASSERT_EQ(-1, send(s, buf, len, 0), NULL);
	T_ASSERT_EQ(ESRCH, errno, NULL);
}
