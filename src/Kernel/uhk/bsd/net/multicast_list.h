/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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
#ifndef _NET_MULTICAST_LIST_H
#define _NET_MULTICAST_LIST_H

#include <sys/queue.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <net/if.h>
#include <net/kpi_interface.h>

/*
 * multicast_util.h:
 * - keep track of multicast addresses on one device for programming on
 *   another (VLAN, BOND)
 */
struct multicast_entry {
	SLIST_ENTRY(multicast_entry)    mc_entries;
	ifmultiaddr_t                   mc_ifma;
};
SLIST_HEAD(multicast_list, multicast_entry);

void
multicast_list_init(struct multicast_list * mc_list);

int
multicast_list_program(struct multicast_list * mc_list,
    struct ifnet * source_ifp,
    struct ifnet * target_ifp);
int
multicast_list_remove(struct multicast_list * mc_list);

#endif /* _NET_MULTICAST_LIST_H */
