/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 26, 2024.
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
 * arp.h
 */

#ifndef _S_ARP_H
#define _S_ARP_H

/*
 * Modification History:
 *
 * 25 Feb 1998	Dieter Siegmund (dieter@apple.com)
 * - created
 */ 

#define ARP_RETURN_SUCCESS			0
#define ARP_RETURN_FAILURE			1
#define ARP_RETURN_INTERNAL_ERROR		2
#define ARP_RETURN_WRITE_FAILED			3
#define ARP_RETURN_READ_FAILED			4
#define ARP_RETURN_HOST_NOT_FOUND		5
#define ARP_RETURN_LAST				6

#include <net/route.h>

typedef struct {
	struct	rt_msghdr m_rtm;
	char	m_space[512];
} route_msg;

int		arp_get(int s, route_msg * msg_p, struct in_addr iaddr,
			int if_index);
int 		arp_delete(int s, struct in_addr iaddr, int if_index);
int		arp_flush(int s, int all, int if_index);
int		arp_open_routing_socket(void);
int		arp_get_next_seq(void);

#endif /* _S_ARP_H */
