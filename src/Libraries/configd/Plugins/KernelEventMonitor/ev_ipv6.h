/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
 * - initial revision
 */


#ifndef _EV_IPV6_H
#define _EV_IPV6_H

#include <netinet6/in6_var.h>

__BEGIN_DECLS

void	interface_update_ipv6(struct ifaddrs *ifap, const char *if_name);
void	ipv6_duplicated_address(const char * if_name, const struct in6_addr * addr,
				int hw_len, const void * hw_addr);
void	nat64_prefix_request(const char *if_name);
void	ipv6_router_expired(const char *if_name);

__END_DECLS

#endif /* _EV_IPV6_H */

