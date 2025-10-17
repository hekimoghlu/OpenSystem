/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
#ifndef __UAPI_TC_CT_H
#define __UAPI_TC_CT_H
#include <linux/types.h>
#include <linux/pkt_cls.h>
enum {
  TCA_CT_UNSPEC,
  TCA_CT_PARMS,
  TCA_CT_TM,
  TCA_CT_ACTION,
  TCA_CT_ZONE,
  TCA_CT_MARK,
  TCA_CT_MARK_MASK,
  TCA_CT_LABELS,
  TCA_CT_LABELS_MASK,
  TCA_CT_NAT_IPV4_MIN,
  TCA_CT_NAT_IPV4_MAX,
  TCA_CT_NAT_IPV6_MIN,
  TCA_CT_NAT_IPV6_MAX,
  TCA_CT_NAT_PORT_MIN,
  TCA_CT_NAT_PORT_MAX,
  TCA_CT_PAD,
  TCA_CT_HELPER_NAME,
  TCA_CT_HELPER_FAMILY,
  TCA_CT_HELPER_PROTO,
  __TCA_CT_MAX
};
#define TCA_CT_MAX (__TCA_CT_MAX - 1)
#define TCA_CT_ACT_COMMIT (1 << 0)
#define TCA_CT_ACT_FORCE (1 << 1)
#define TCA_CT_ACT_CLEAR (1 << 2)
#define TCA_CT_ACT_NAT (1 << 3)
#define TCA_CT_ACT_NAT_SRC (1 << 4)
#define TCA_CT_ACT_NAT_DST (1 << 5)
struct tc_ct {
  tc_gen;
};
#endif
