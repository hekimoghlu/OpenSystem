/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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
#ifndef _IPT_ECN_H
#define _IPT_ECN_H
#include <linux/netfilter/xt_ecn.h>
#define ipt_ecn_info xt_ecn_info
enum {
  IPT_ECN_IP_MASK = XT_ECN_IP_MASK,
  IPT_ECN_OP_MATCH_IP = XT_ECN_OP_MATCH_IP,
  IPT_ECN_OP_MATCH_ECE = XT_ECN_OP_MATCH_ECE,
  IPT_ECN_OP_MATCH_CWR = XT_ECN_OP_MATCH_CWR,
  IPT_ECN_OP_MATCH_MASK = XT_ECN_OP_MATCH_MASK,
};
#endif
