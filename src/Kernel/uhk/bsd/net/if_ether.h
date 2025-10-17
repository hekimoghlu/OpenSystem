/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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
#ifndef _NET_IF_ETHER_H
#define _NET_IF_ETHER_H
#ifdef KERNEL

#include <net/kpi_interface.h>

__BEGIN_DECLS
/* Not exported */
extern int ether_family_init(void);

/*
 * These functions may be used for an interface emulating an ethernet
 * interface and not using IOKit. If you use IOKit and the IOKit
 * Ethernet Family, these functions will be set for you. Use these
 * functions when filling out the ifnet_init_params structure.
 */
errno_t ether_demux(ifnet_t interface, mbuf_t packet, char* header,
    protocol_family_t *protocol);
errno_t ether_add_proto(ifnet_t interface, protocol_family_t protocol,
    const struct ifnet_demux_desc *demux_list __counted_by(demux_count), u_int32_t demux_count);
errno_t ether_del_proto(ifnet_t interface, protocol_family_t protocol);
#if KPI_INTERFACE_EMBEDDED
errno_t ether_frameout(ifnet_t interface, mbuf_t *packet,
    const struct sockaddr *dest, IFNET_LLADDR_T dest_lladdr,
    IFNET_FRAME_TYPE_T frame_type,
    u_int32_t *prepend_len, u_int32_t *postpend_len);
#else /* !KPI_INTERFACE_EMBEDDED */
errno_t ether_frameout(ifnet_t interface, mbuf_t *packet,
    const struct sockaddr *dest, IFNET_LLADDR_T dest_lladdr,
    IFNET_FRAME_TYPE_T frame_type);
#endif /* !KPI_INTERFACE_EMBEDDED */
#ifdef KERNEL_PRIVATE
errno_t ether_frameout_extended(ifnet_t interface, mbuf_t *packet,
    const struct sockaddr *dest, IFNET_LLADDR_T dest_lladdr,
    IFNET_FRAME_TYPE_T frame_type,
    u_int32_t *prepend_len, u_int32_t *postpend_len);
#endif /* KERNEL_PRIVATE */
errno_t ether_ioctl(ifnet_t interface, u_int32_t command, void* data);
errno_t ether_check_multi(ifnet_t ifp, const struct sockaddr *multicast);

__END_DECLS

#endif /* KERNEL */
#endif /* _NET_IF_ETHER_H */
