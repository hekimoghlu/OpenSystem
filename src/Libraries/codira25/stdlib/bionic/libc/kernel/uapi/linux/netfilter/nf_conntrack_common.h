/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#ifndef _UAPI_NF_CONNTRACK_COMMON_H
#define _UAPI_NF_CONNTRACK_COMMON_H
enum ip_conntrack_info {
  IP_CT_ESTABLISHED,
  IP_CT_RELATED,
  IP_CT_NEW,
  IP_CT_IS_REPLY,
  IP_CT_ESTABLISHED_REPLY = IP_CT_ESTABLISHED + IP_CT_IS_REPLY,
  IP_CT_RELATED_REPLY = IP_CT_RELATED + IP_CT_IS_REPLY,
  IP_CT_NUMBER,
  IP_CT_NEW_REPLY = IP_CT_NUMBER,
};
#define NF_CT_STATE_INVALID_BIT (1 << 0)
#define NF_CT_STATE_BIT(ctinfo) (1 << ((ctinfo) % IP_CT_IS_REPLY + 1))
#define NF_CT_STATE_UNTRACKED_BIT (1 << 6)
enum ip_conntrack_status {
  IPS_EXPECTED_BIT = 0,
  IPS_EXPECTED = (1 << IPS_EXPECTED_BIT),
  IPS_SEEN_REPLY_BIT = 1,
  IPS_SEEN_REPLY = (1 << IPS_SEEN_REPLY_BIT),
  IPS_ASSURED_BIT = 2,
  IPS_ASSURED = (1 << IPS_ASSURED_BIT),
  IPS_CONFIRMED_BIT = 3,
  IPS_CONFIRMED = (1 << IPS_CONFIRMED_BIT),
  IPS_SRC_NAT_BIT = 4,
  IPS_SRC_NAT = (1 << IPS_SRC_NAT_BIT),
  IPS_DST_NAT_BIT = 5,
  IPS_DST_NAT = (1 << IPS_DST_NAT_BIT),
  IPS_NAT_MASK = (IPS_DST_NAT | IPS_SRC_NAT),
  IPS_SEQ_ADJUST_BIT = 6,
  IPS_SEQ_ADJUST = (1 << IPS_SEQ_ADJUST_BIT),
  IPS_SRC_NAT_DONE_BIT = 7,
  IPS_SRC_NAT_DONE = (1 << IPS_SRC_NAT_DONE_BIT),
  IPS_DST_NAT_DONE_BIT = 8,
  IPS_DST_NAT_DONE = (1 << IPS_DST_NAT_DONE_BIT),
  IPS_NAT_DONE_MASK = (IPS_DST_NAT_DONE | IPS_SRC_NAT_DONE),
  IPS_DYING_BIT = 9,
  IPS_DYING = (1 << IPS_DYING_BIT),
  IPS_FIXED_TIMEOUT_BIT = 10,
  IPS_FIXED_TIMEOUT = (1 << IPS_FIXED_TIMEOUT_BIT),
  IPS_TEMPLATE_BIT = 11,
  IPS_TEMPLATE = (1 << IPS_TEMPLATE_BIT),
  IPS_UNTRACKED_BIT = 12,
  IPS_UNTRACKED = (1 << IPS_UNTRACKED_BIT),
  IPS_HELPER_BIT = 13,
  IPS_HELPER = (1 << IPS_HELPER_BIT),
  IPS_OFFLOAD_BIT = 14,
  IPS_OFFLOAD = (1 << IPS_OFFLOAD_BIT),
  IPS_HW_OFFLOAD_BIT = 15,
  IPS_HW_OFFLOAD = (1 << IPS_HW_OFFLOAD_BIT),
  IPS_UNCHANGEABLE_MASK = (IPS_NAT_DONE_MASK | IPS_NAT_MASK | IPS_EXPECTED | IPS_CONFIRMED | IPS_DYING | IPS_SEQ_ADJUST | IPS_TEMPLATE | IPS_UNTRACKED | IPS_OFFLOAD | IPS_HW_OFFLOAD),
  __IPS_MAX_BIT = 16,
};
enum ip_conntrack_events {
  IPCT_NEW,
  IPCT_RELATED,
  IPCT_DESTROY,
  IPCT_REPLY,
  IPCT_ASSURED,
  IPCT_PROTOINFO,
  IPCT_HELPER,
  IPCT_MARK,
  IPCT_SEQADJ,
  IPCT_NATSEQADJ = IPCT_SEQADJ,
  IPCT_SECMARK,
  IPCT_LABEL,
  IPCT_SYNPROXY,
};
enum ip_conntrack_expect_events {
  IPEXP_NEW,
  IPEXP_DESTROY,
};
#define NF_CT_EXPECT_PERMANENT 0x1
#define NF_CT_EXPECT_INACTIVE 0x2
#define NF_CT_EXPECT_USERSPACE 0x4
#endif
