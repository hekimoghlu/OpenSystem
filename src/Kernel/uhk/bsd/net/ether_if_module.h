/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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
#ifndef _NET_ETHER_IF_MODULE_H
#define _NET_ETHER_IF_MODULE_H

#include <net/dlil.h>

extern errno_t ether_attach_inet(ifnet_t ifp, protocol_family_t protocol_family);
extern void ether_detach_inet(ifnet_t ifp, protocol_family_t protocol_family);
extern errno_t ether_attach_inet6(ifnet_t ifp, protocol_family_t protocol_family);
extern void ether_detach_inet6(ifnet_t ifp, protocol_family_t protocol_family);
extern errno_t ether_attach_at(struct ifnet *ifp, protocol_family_t proto_family);
extern void ether_detach_at(struct ifnet *ifp, protocol_family_t proto_family);

#endif /* _NET_ETHER_IF_MODULE_H */
