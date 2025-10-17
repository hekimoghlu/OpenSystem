/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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
#ifndef __LINUX_TC_MPLS_H
#define __LINUX_TC_MPLS_H
#include <linux/pkt_cls.h>
#define TCA_MPLS_ACT_POP 1
#define TCA_MPLS_ACT_PUSH 2
#define TCA_MPLS_ACT_MODIFY 3
#define TCA_MPLS_ACT_DEC_TTL 4
#define TCA_MPLS_ACT_MAC_PUSH 5
struct tc_mpls {
  tc_gen;
  int m_action;
};
enum {
  TCA_MPLS_UNSPEC,
  TCA_MPLS_TM,
  TCA_MPLS_PARMS,
  TCA_MPLS_PAD,
  TCA_MPLS_PROTO,
  TCA_MPLS_LABEL,
  TCA_MPLS_TC,
  TCA_MPLS_TTL,
  TCA_MPLS_BOS,
  __TCA_MPLS_MAX,
};
#define TCA_MPLS_MAX (__TCA_MPLS_MAX - 1)
#endif
