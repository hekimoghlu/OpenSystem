/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#ifndef __LINUX_BRIDGE_EBT_AMONG_H
#define __LINUX_BRIDGE_EBT_AMONG_H
#include <linux/types.h>
#define EBT_AMONG_DST 0x01
#define EBT_AMONG_SRC 0x02
struct ebt_mac_wormhash_tuple {
  __u32 cmp[2];
  __be32 ip;
};
struct ebt_mac_wormhash {
  int table[257];
  int poolsize;
  struct ebt_mac_wormhash_tuple pool[];
};
#define ebt_mac_wormhash_size(x) ((x) ? sizeof(struct ebt_mac_wormhash) + (x)->poolsize * sizeof(struct ebt_mac_wormhash_tuple) : 0)
struct ebt_among_info {
  int wh_dst_ofs;
  int wh_src_ofs;
  int bitmask;
};
#define EBT_AMONG_DST_NEG 0x1
#define EBT_AMONG_SRC_NEG 0x2
#define ebt_among_wh_dst(x) ((x)->wh_dst_ofs ? (struct ebt_mac_wormhash *) ((char *) (x) + (x)->wh_dst_ofs) : NULL)
#define ebt_among_wh_src(x) ((x)->wh_src_ofs ? (struct ebt_mac_wormhash *) ((char *) (x) + (x)->wh_src_ofs) : NULL)
#define EBT_AMONG_MATCH "among"
#endif
