/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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
#ifndef _NF_CONNTRACK_TUPLE_COMMON_H
#define _NF_CONNTRACK_TUPLE_COMMON_H
#include <linux/types.h>
#include <linux/netfilter.h>
#include <linux/netfilter/nf_conntrack_common.h>
enum ip_conntrack_dir {
  IP_CT_DIR_ORIGINAL,
  IP_CT_DIR_REPLY,
  IP_CT_DIR_MAX
};
union nf_conntrack_man_proto {
  __be16 all;
  struct {
    __be16 port;
  } tcp;
  struct {
    __be16 port;
  } udp;
  struct {
    __be16 id;
  } icmp;
  struct {
    __be16 port;
  } dccp;
  struct {
    __be16 port;
  } sctp;
  struct {
    __be16 key;
  } gre;
};
#define CTINFO2DIR(ctinfo) ((ctinfo) >= IP_CT_IS_REPLY ? IP_CT_DIR_REPLY : IP_CT_DIR_ORIGINAL)
#endif
