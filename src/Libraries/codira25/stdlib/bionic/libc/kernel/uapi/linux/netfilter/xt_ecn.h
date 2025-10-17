/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#ifndef _XT_ECN_H
#define _XT_ECN_H
#include <linux/types.h>
#include <linux/netfilter/xt_dscp.h>
#define XT_ECN_IP_MASK (~XT_DSCP_MASK)
#define XT_ECN_OP_MATCH_IP 0x01
#define XT_ECN_OP_MATCH_ECE 0x10
#define XT_ECN_OP_MATCH_CWR 0x20
#define XT_ECN_OP_MATCH_MASK 0xce
struct xt_ecn_info {
  __u8 operation;
  __u8 invert;
  __u8 ip_ect;
  union {
    struct {
      __u8 ect;
    } tcp;
  } proto;
};
#endif
