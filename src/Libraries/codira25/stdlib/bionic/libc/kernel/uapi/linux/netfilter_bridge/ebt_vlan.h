/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#ifndef __LINUX_BRIDGE_EBT_VLAN_H
#define __LINUX_BRIDGE_EBT_VLAN_H
#include <linux/types.h>
#define EBT_VLAN_ID 0x01
#define EBT_VLAN_PRIO 0x02
#define EBT_VLAN_ENCAP 0x04
#define EBT_VLAN_MASK (EBT_VLAN_ID | EBT_VLAN_PRIO | EBT_VLAN_ENCAP)
#define EBT_VLAN_MATCH "vlan"
struct ebt_vlan_info {
  __u16 id;
  __u8 prio;
  __be16 encap;
  __u8 bitmask;
  __u8 invflags;
};
#endif
