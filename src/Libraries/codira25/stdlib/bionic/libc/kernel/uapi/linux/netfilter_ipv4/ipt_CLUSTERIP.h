/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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
#ifndef _IPT_CLUSTERIP_H_target
#define _IPT_CLUSTERIP_H_target
#include <linux/types.h>
#include <linux/if_ether.h>
enum clusterip_hashmode {
  CLUSTERIP_HASHMODE_SIP = 0,
  CLUSTERIP_HASHMODE_SIP_SPT,
  CLUSTERIP_HASHMODE_SIP_SPT_DPT,
};
#define CLUSTERIP_HASHMODE_MAX CLUSTERIP_HASHMODE_SIP_SPT_DPT
#define CLUSTERIP_MAX_NODES 16
#define CLUSTERIP_FLAG_NEW 0x00000001
struct clusterip_config;
struct ipt_clusterip_tgt_info {
  __u32 flags;
  __u8 clustermac[ETH_ALEN];
  __u16 num_total_nodes;
  __u16 num_local_nodes;
  __u16 local_nodes[CLUSTERIP_MAX_NODES];
  __u32 hash_mode;
  __u32 hash_initval;
  struct clusterip_config * config;
};
#endif
