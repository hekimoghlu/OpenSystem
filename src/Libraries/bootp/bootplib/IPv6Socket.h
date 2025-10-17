/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
 * IPv6Socket.h
 * - common functions for creating/sending packets over IPv6 sockets
 */

/* 
 * Modification History
 *
 * May 24, 2013		Dieter Siegmund (dieter@apple.com)
 * - created
 */

#ifndef _S_IPV6SOCKET_H
#define _S_IPV6SOCKET_H

#include <stdbool.h>
#include <netinet/in.h>
#include <netinet/ip6.h>

int
IPv6SocketSend(int sockfd, int ifindex, const struct sockaddr_in6 * dest,
	       const void * pkt, int pkt_size, int hlim);

#endif /* _S_IPV6SOCKET_H */
