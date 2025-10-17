/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
 * Modification History
 *
 * August 5, 2002	Allan Nathanson <ajn@apple.com>
 * - split code out from eventmon.c
 */


#ifndef _EV_IPV4_H
#define _EV_IPV4_H

#include <TargetConditionals.h>
#include <netinet/in_var.h>

__BEGIN_DECLS

void	ipv4_interface_update(struct ifaddrs *ifap, const char *if_name);

void	ipv4_arp_collision(const char *if_name,
			   struct in_addr ip_addr,
			   int hw_len, const void * hw_addr);

#if	!TARGET_OS_IPHONE
void	ipv4_port_in_use(uint16_t port, pid_t req_pid);
#endif	/* !TARGET_OS_IPHONE */

void	ipv4_router_arp_failure(const char * if_name);
void	ipv4_router_arp_alive(const char * if_name);

__END_DECLS

#endif /* _EV_IPV4_H */

