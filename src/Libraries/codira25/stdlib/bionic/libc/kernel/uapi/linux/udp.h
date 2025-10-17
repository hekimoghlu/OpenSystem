/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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
#ifndef _UAPI_LINUX_UDP_H
#define _UAPI_LINUX_UDP_H
#include <linux/types.h>
struct __kernel_udphdr {
  __be16 source;
  __be16 dest;
  __be16 len;
  __sum16 check;
};
#define UDP_CORK 1
#define UDP_ENCAP 100
#define UDP_NO_CHECK6_TX 101
#define UDP_NO_CHECK6_RX 102
#define UDP_SEGMENT 103
#define UDP_GRO 104
#define UDP_ENCAP_ESPINUDP_NON_IKE 1
#define UDP_ENCAP_ESPINUDP 2
#define UDP_ENCAP_L2TPINUDP 3
#define UDP_ENCAP_GTP0 4
#define UDP_ENCAP_GTP1U 5
#define UDP_ENCAP_RXRPC 6
#define TCP_ENCAP_ESPINTCP 7
#endif
