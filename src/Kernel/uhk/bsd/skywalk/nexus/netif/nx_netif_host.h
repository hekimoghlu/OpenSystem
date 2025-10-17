/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#ifndef _SKYWALK_NEXUS_NETIF_HOST_H_
#define _SKYWALK_NEXUS_NETIF_HOST_H_

#include <skywalk/os_skywalk_private.h>

__BEGIN_DECLS
extern int nx_netif_host_na_activate(struct nexus_adapter *,
    na_activate_mode_t);
extern int nx_netif_host_krings_create(struct nexus_adapter *,
    struct kern_channel *);
extern void nx_netif_host_krings_delete(struct nexus_adapter *,
    struct kern_channel *, boolean_t);
extern int nx_netif_host_na_rxsync(struct __kern_channel_ring *,
    struct proc *, uint32_t);
extern int nx_netif_host_na_txsync(struct __kern_channel_ring *,
    struct proc *, uint32_t);
extern int nx_netif_host_na_special(struct nexus_adapter *,
    struct kern_channel *, struct chreq *, nxspec_cmd_t);
extern int nx_netif_host_output(struct ifnet *, struct mbuf *);
extern boolean_t netif_chain_enqueue_enabled(struct ifnet *);
__END_DECLS
#endif /* _SKYWALK_NEXUS_NETIF_HOST_H_ */
