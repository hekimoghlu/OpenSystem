/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 28, 2022.
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
#ifndef __LINUX_TC_VLAN_H
#define __LINUX_TC_VLAN_H
#include <linux/pkt_cls.h>
#define TCA_VLAN_ACT_POP 1
#define TCA_VLAN_ACT_PUSH 2
#define TCA_VLAN_ACT_MODIFY 3
#define TCA_VLAN_ACT_POP_ETH 4
#define TCA_VLAN_ACT_PUSH_ETH 5
struct tc_vlan {
  tc_gen;
  int v_action;
};
enum {
  TCA_VLAN_UNSPEC,
  TCA_VLAN_TM,
  TCA_VLAN_PARMS,
  TCA_VLAN_PUSH_VLAN_ID,
  TCA_VLAN_PUSH_VLAN_PROTOCOL,
  TCA_VLAN_PAD,
  TCA_VLAN_PUSH_VLAN_PRIORITY,
  TCA_VLAN_PUSH_ETH_DST,
  TCA_VLAN_PUSH_ETH_SRC,
  __TCA_VLAN_MAX,
};
#define TCA_VLAN_MAX (__TCA_VLAN_MAX - 1)
#endif
