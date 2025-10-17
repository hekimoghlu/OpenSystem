/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
#ifndef _XT_SET_H
#define _XT_SET_H
#include <linux/types.h>
#include <linux/netfilter/ipset/ip_set.h>
#define IPSET_SRC 0x01
#define IPSET_DST 0x02
#define IPSET_MATCH_INV 0x04
struct xt_set_info_v0 {
  ip_set_id_t index;
  union {
    __u32 flags[IPSET_DIM_MAX + 1];
    struct {
      __u32 __flags[IPSET_DIM_MAX];
      __u8 dim;
      __u8 flags;
    } compat;
  } u;
};
struct xt_set_info_match_v0 {
  struct xt_set_info_v0 match_set;
};
struct xt_set_info_target_v0 {
  struct xt_set_info_v0 add_set;
  struct xt_set_info_v0 del_set;
};
struct xt_set_info {
  ip_set_id_t index;
  __u8 dim;
  __u8 flags;
};
struct xt_set_info_match_v1 {
  struct xt_set_info match_set;
};
struct xt_set_info_target_v1 {
  struct xt_set_info add_set;
  struct xt_set_info del_set;
};
struct xt_set_info_target_v2 {
  struct xt_set_info add_set;
  struct xt_set_info del_set;
  __u32 flags;
  __u32 timeout;
};
struct xt_set_info_match_v3 {
  struct xt_set_info match_set;
  struct ip_set_counter_match0 packets;
  struct ip_set_counter_match0 bytes;
  __u32 flags;
};
struct xt_set_info_target_v3 {
  struct xt_set_info add_set;
  struct xt_set_info del_set;
  struct xt_set_info map_set;
  __u32 flags;
  __u32 timeout;
};
struct xt_set_info_match_v4 {
  struct xt_set_info match_set;
  struct ip_set_counter_match packets;
  struct ip_set_counter_match bytes;
  __u32 flags;
};
#endif
