/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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
#ifndef _XT_TCPUDP_H
#define _XT_TCPUDP_H
#include <linux/types.h>
struct xt_tcp {
  __u16 spts[2];
  __u16 dpts[2];
  __u8 option;
  __u8 flg_mask;
  __u8 flg_cmp;
  __u8 invflags;
};
#define XT_TCP_INV_SRCPT 0x01
#define XT_TCP_INV_DSTPT 0x02
#define XT_TCP_INV_FLAGS 0x04
#define XT_TCP_INV_OPTION 0x08
#define XT_TCP_INV_MASK 0x0F
struct xt_udp {
  __u16 spts[2];
  __u16 dpts[2];
  __u8 invflags;
};
#define XT_UDP_INV_SRCPT 0x01
#define XT_UDP_INV_DSTPT 0x02
#define XT_UDP_INV_MASK 0x03
#endif
