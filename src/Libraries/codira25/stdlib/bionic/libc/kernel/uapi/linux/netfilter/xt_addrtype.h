/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#ifndef _XT_ADDRTYPE_H
#define _XT_ADDRTYPE_H
#include <linux/types.h>
enum {
  XT_ADDRTYPE_INVERT_SOURCE = 0x0001,
  XT_ADDRTYPE_INVERT_DEST = 0x0002,
  XT_ADDRTYPE_LIMIT_IFACE_IN = 0x0004,
  XT_ADDRTYPE_LIMIT_IFACE_OUT = 0x0008,
};
enum {
  XT_ADDRTYPE_UNSPEC = 1 << 0,
  XT_ADDRTYPE_UNICAST = 1 << 1,
  XT_ADDRTYPE_LOCAL = 1 << 2,
  XT_ADDRTYPE_BROADCAST = 1 << 3,
  XT_ADDRTYPE_ANYCAST = 1 << 4,
  XT_ADDRTYPE_MULTICAST = 1 << 5,
  XT_ADDRTYPE_BLACKHOLE = 1 << 6,
  XT_ADDRTYPE_UNREACHABLE = 1 << 7,
  XT_ADDRTYPE_PROHIBIT = 1 << 8,
  XT_ADDRTYPE_THROW = 1 << 9,
  XT_ADDRTYPE_NAT = 1 << 10,
  XT_ADDRTYPE_XRESOLVE = 1 << 11,
};
struct xt_addrtype_info_v1 {
  __u16 source;
  __u16 dest;
  __u32 flags;
};
struct xt_addrtype_info {
  __u16 source;
  __u16 dest;
  __u32 invert_source;
  __u32 invert_dest;
};
#endif
