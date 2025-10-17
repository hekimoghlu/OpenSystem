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
#ifndef _IP6T_SRH_H
#define _IP6T_SRH_H
#include <linux/types.h>
#include <linux/netfilter.h>
#define IP6T_SRH_NEXTHDR 0x0001
#define IP6T_SRH_LEN_EQ 0x0002
#define IP6T_SRH_LEN_GT 0x0004
#define IP6T_SRH_LEN_LT 0x0008
#define IP6T_SRH_SEGS_EQ 0x0010
#define IP6T_SRH_SEGS_GT 0x0020
#define IP6T_SRH_SEGS_LT 0x0040
#define IP6T_SRH_LAST_EQ 0x0080
#define IP6T_SRH_LAST_GT 0x0100
#define IP6T_SRH_LAST_LT 0x0200
#define IP6T_SRH_TAG 0x0400
#define IP6T_SRH_PSID 0x0800
#define IP6T_SRH_NSID 0x1000
#define IP6T_SRH_LSID 0x2000
#define IP6T_SRH_MASK 0x3FFF
#define IP6T_SRH_INV_NEXTHDR 0x0001
#define IP6T_SRH_INV_LEN_EQ 0x0002
#define IP6T_SRH_INV_LEN_GT 0x0004
#define IP6T_SRH_INV_LEN_LT 0x0008
#define IP6T_SRH_INV_SEGS_EQ 0x0010
#define IP6T_SRH_INV_SEGS_GT 0x0020
#define IP6T_SRH_INV_SEGS_LT 0x0040
#define IP6T_SRH_INV_LAST_EQ 0x0080
#define IP6T_SRH_INV_LAST_GT 0x0100
#define IP6T_SRH_INV_LAST_LT 0x0200
#define IP6T_SRH_INV_TAG 0x0400
#define IP6T_SRH_INV_PSID 0x0800
#define IP6T_SRH_INV_NSID 0x1000
#define IP6T_SRH_INV_LSID 0x2000
#define IP6T_SRH_INV_MASK 0x3FFF
struct ip6t_srh {
  __u8 next_hdr;
  __u8 hdr_len;
  __u8 segs_left;
  __u8 last_entry;
  __u16 tag;
  __u16 mt_flags;
  __u16 mt_invflags;
};
struct ip6t_srh1 {
  __u8 next_hdr;
  __u8 hdr_len;
  __u8 segs_left;
  __u8 last_entry;
  __u16 tag;
  struct in6_addr psid_addr;
  struct in6_addr nsid_addr;
  struct in6_addr lsid_addr;
  struct in6_addr psid_msk;
  struct in6_addr nsid_msk;
  struct in6_addr lsid_msk;
  __u16 mt_flags;
  __u16 mt_invflags;
};
#endif
