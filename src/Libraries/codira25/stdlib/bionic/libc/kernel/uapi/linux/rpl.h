/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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
#ifndef _UAPI_LINUX_RPL_H
#define _UAPI_LINUX_RPL_H
#include <asm/byteorder.h>
#include <linux/types.h>
#include <linux/in6.h>
struct ipv6_rpl_sr_hdr {
  __u8 nexthdr;
  __u8 hdrlen;
  __u8 type;
  __u8 segments_left;
#ifdef __LITTLE_ENDIAN_BITFIELD
  __u32 cmpre : 4, cmpri : 4, reserved : 4, pad : 4, reserved1 : 16;
#elif defined(__BIG_ENDIAN_BITFIELD)
  __u32 cmpri : 4, cmpre : 4, pad : 4, reserved : 20;
#else
#error "Please fix <asm/byteorder.h>"
#endif
  union {
    __DECLARE_FLEX_ARRAY(struct in6_addr, addr);
    __DECLARE_FLEX_ARRAY(__u8, data);
  } segments;
} __attribute__((packed));
#define rpl_segaddr segments.addr
#define rpl_segdata segments.data
#endif
