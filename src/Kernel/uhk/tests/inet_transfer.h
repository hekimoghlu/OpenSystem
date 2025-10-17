/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
 * inet_transfer.h
 * - perform IPv4/IPv6 UDP/TCP transfer tests
 */

#ifndef _S_INET_TRANSFER_H
#define _S_INET_TRANSFER_H

#include <stdint.h>
#include <netinet/in.h>

static inline const char *
af_get_str(uint8_t af)
{
	return (af == AF_INET) ? "IPv4" : "IPv6";
}

static inline const char *
ipproto_get_str(uint8_t proto)
{
	const char * str;

	switch (proto) {
	case IPPROTO_UDP:
		str = "UDP";
		break;
	case IPPROTO_TCP:
		str = "TCP";
		break;
	default:
		str = "<?>";
		break;
	}
	return str;
}

typedef union {
	struct in_addr  v4;
	struct in6_addr v6;
} inet_address, *inet_address_t;

typedef struct {
	uint8_t         af;
	uint8_t         proto;
	uint16_t        port;
	inet_address    addr;
} inet_endpoint, *inet_endpoint_t;

bool
inet_transfer_local(inet_endpoint_t server_endpoint,
    int server_if_index,
    int client_if_index);

void
inet_test_traffic(uint8_t af, inet_address_t server,
    const char * server_ifname, int server_if_index,
    const char * client_ifname, int client_if_index);

const char *
inet_transfer_error_string(void);

#endif /* _S_INET_TRANSFER_H */
