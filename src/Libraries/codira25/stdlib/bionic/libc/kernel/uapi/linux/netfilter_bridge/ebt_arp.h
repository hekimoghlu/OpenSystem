/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#ifndef __LINUX_BRIDGE_EBT_ARP_H
#define __LINUX_BRIDGE_EBT_ARP_H
#include <linux/types.h>
#include <linux/if_ether.h>
#define EBT_ARP_OPCODE 0x01
#define EBT_ARP_HTYPE 0x02
#define EBT_ARP_PTYPE 0x04
#define EBT_ARP_SRC_IP 0x08
#define EBT_ARP_DST_IP 0x10
#define EBT_ARP_SRC_MAC 0x20
#define EBT_ARP_DST_MAC 0x40
#define EBT_ARP_GRAT 0x80
#define EBT_ARP_MASK (EBT_ARP_OPCODE | EBT_ARP_HTYPE | EBT_ARP_PTYPE | EBT_ARP_SRC_IP | EBT_ARP_DST_IP | EBT_ARP_SRC_MAC | EBT_ARP_DST_MAC | EBT_ARP_GRAT)
#define EBT_ARP_MATCH "arp"
struct ebt_arp_info {
  __be16 htype;
  __be16 ptype;
  __be16 opcode;
  __be32 saddr;
  __be32 smsk;
  __be32 daddr;
  __be32 dmsk;
  unsigned char smaddr[ETH_ALEN];
  unsigned char smmsk[ETH_ALEN];
  unsigned char dmaddr[ETH_ALEN];
  unsigned char dmmsk[ETH_ALEN];
  __u8 bitmask;
  __u8 invflags;
};
#endif
