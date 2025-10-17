/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
#ifndef _UAPI__LINUX_BRIDGE_NETFILTER_H
#define _UAPI__LINUX_BRIDGE_NETFILTER_H
#include <linux/in.h>
#include <linux/netfilter.h>
#include <linux/if_ether.h>
#include <linux/if_vlan.h>
#include <linux/if_pppox.h>
#include <limits.h>
#define NF_BR_PRE_ROUTING 0
#define NF_BR_LOCAL_IN 1
#define NF_BR_FORWARD 2
#define NF_BR_LOCAL_OUT 3
#define NF_BR_POST_ROUTING 4
#define NF_BR_BROUTING 5
#define NF_BR_NUMHOOKS 6
enum nf_br_hook_priorities {
  NF_BR_PRI_FIRST = INT_MIN,
  NF_BR_PRI_NAT_DST_BRIDGED = - 300,
  NF_BR_PRI_FILTER_BRIDGED = - 200,
  NF_BR_PRI_BRNF = 0,
  NF_BR_PRI_NAT_DST_OTHER = 100,
  NF_BR_PRI_FILTER_OTHER = 200,
  NF_BR_PRI_NAT_SRC = 300,
  NF_BR_PRI_LAST = INT_MAX,
};
#endif
