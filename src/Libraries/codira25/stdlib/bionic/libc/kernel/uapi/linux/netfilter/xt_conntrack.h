/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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
#ifndef _XT_CONNTRACK_H
#define _XT_CONNTRACK_H
#include <linux/types.h>
#include <linux/netfilter.h>
#include <linux/netfilter/nf_conntrack_tuple_common.h>
#define XT_CONNTRACK_STATE_BIT(ctinfo) (1 << ((ctinfo) % IP_CT_IS_REPLY + 1))
#define XT_CONNTRACK_STATE_INVALID (1 << 0)
#define XT_CONNTRACK_STATE_SNAT (1 << (IP_CT_NUMBER + 1))
#define XT_CONNTRACK_STATE_DNAT (1 << (IP_CT_NUMBER + 2))
#define XT_CONNTRACK_STATE_UNTRACKED (1 << (IP_CT_NUMBER + 3))
enum {
  XT_CONNTRACK_STATE = 1 << 0,
  XT_CONNTRACK_PROTO = 1 << 1,
  XT_CONNTRACK_ORIGSRC = 1 << 2,
  XT_CONNTRACK_ORIGDST = 1 << 3,
  XT_CONNTRACK_REPLSRC = 1 << 4,
  XT_CONNTRACK_REPLDST = 1 << 5,
  XT_CONNTRACK_STATUS = 1 << 6,
  XT_CONNTRACK_EXPIRES = 1 << 7,
  XT_CONNTRACK_ORIGSRC_PORT = 1 << 8,
  XT_CONNTRACK_ORIGDST_PORT = 1 << 9,
  XT_CONNTRACK_REPLSRC_PORT = 1 << 10,
  XT_CONNTRACK_REPLDST_PORT = 1 << 11,
  XT_CONNTRACK_DIRECTION = 1 << 12,
  XT_CONNTRACK_STATE_ALIAS = 1 << 13,
};
struct xt_conntrack_mtinfo1 {
  union nf_inet_addr origsrc_addr, origsrc_mask;
  union nf_inet_addr origdst_addr, origdst_mask;
  union nf_inet_addr replsrc_addr, replsrc_mask;
  union nf_inet_addr repldst_addr, repldst_mask;
  __u32 expires_min, expires_max;
  __u16 l4proto;
  __be16 origsrc_port, origdst_port;
  __be16 replsrc_port, repldst_port;
  __u16 match_flags, invert_flags;
  __u8 state_mask, status_mask;
};
struct xt_conntrack_mtinfo2 {
  union nf_inet_addr origsrc_addr, origsrc_mask;
  union nf_inet_addr origdst_addr, origdst_mask;
  union nf_inet_addr replsrc_addr, replsrc_mask;
  union nf_inet_addr repldst_addr, repldst_mask;
  __u32 expires_min, expires_max;
  __u16 l4proto;
  __be16 origsrc_port, origdst_port;
  __be16 replsrc_port, repldst_port;
  __u16 match_flags, invert_flags;
  __u16 state_mask, status_mask;
};
struct xt_conntrack_mtinfo3 {
  union nf_inet_addr origsrc_addr, origsrc_mask;
  union nf_inet_addr origdst_addr, origdst_mask;
  union nf_inet_addr replsrc_addr, replsrc_mask;
  union nf_inet_addr repldst_addr, repldst_mask;
  __u32 expires_min, expires_max;
  __u16 l4proto;
  __u16 origsrc_port, origdst_port;
  __u16 replsrc_port, repldst_port;
  __u16 match_flags, invert_flags;
  __u16 state_mask, status_mask;
  __u16 origsrc_port_high, origdst_port_high;
  __u16 replsrc_port_high, repldst_port_high;
};
#endif
