/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
#ifndef _IP6_TUNNEL_H
#define _IP6_TUNNEL_H
#include <linux/types.h>
#include <linux/if.h>
#include <linux/in6.h>
#define IPV6_TLV_TNL_ENCAP_LIMIT 4
#define IPV6_DEFAULT_TNL_ENCAP_LIMIT 4
#define IP6_TNL_F_IGN_ENCAP_LIMIT 0x1
#define IP6_TNL_F_USE_ORIG_TCLASS 0x2
#define IP6_TNL_F_USE_ORIG_FLOWLABEL 0x4
#define IP6_TNL_F_MIP6_DEV 0x8
#define IP6_TNL_F_RCV_DSCP_COPY 0x10
#define IP6_TNL_F_USE_ORIG_FWMARK 0x20
#define IP6_TNL_F_ALLOW_LOCAL_REMOTE 0x40
struct ip6_tnl_parm {
  char name[IFNAMSIZ];
  int link;
  __u8 proto;
  __u8 encap_limit;
  __u8 hop_limit;
  __be32 flowinfo;
  __u32 flags;
  struct in6_addr laddr;
  struct in6_addr raddr;
};
struct ip6_tnl_parm2 {
  char name[IFNAMSIZ];
  int link;
  __u8 proto;
  __u8 encap_limit;
  __u8 hop_limit;
  __be32 flowinfo;
  __u32 flags;
  struct in6_addr laddr;
  struct in6_addr raddr;
  __be16 i_flags;
  __be16 o_flags;
  __be32 i_key;
  __be32 o_key;
};
#endif
