/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
#ifndef XT_HMARK_H_
#define XT_HMARK_H_
#include <linux/types.h>
#include <linux/netfilter.h>
enum {
  XT_HMARK_SADDR_MASK,
  XT_HMARK_DADDR_MASK,
  XT_HMARK_SPI,
  XT_HMARK_SPI_MASK,
  XT_HMARK_SPORT,
  XT_HMARK_DPORT,
  XT_HMARK_SPORT_MASK,
  XT_HMARK_DPORT_MASK,
  XT_HMARK_PROTO_MASK,
  XT_HMARK_RND,
  XT_HMARK_MODULUS,
  XT_HMARK_OFFSET,
  XT_HMARK_CT,
  XT_HMARK_METHOD_L3,
  XT_HMARK_METHOD_L3_4,
};
#define XT_HMARK_FLAG(flag) (1 << flag)
union hmark_ports {
  struct {
    __u16 src;
    __u16 dst;
  } p16;
  struct {
    __be16 src;
    __be16 dst;
  } b16;
  __u32 v32;
  __be32 b32;
};
struct xt_hmark_info {
  union nf_inet_addr src_mask;
  union nf_inet_addr dst_mask;
  union hmark_ports port_mask;
  union hmark_ports port_set;
  __u32 flags;
  __u16 proto_mask;
  __u32 hashrnd;
  __u32 hmodulus;
  __u32 hoffset;
};
#endif
