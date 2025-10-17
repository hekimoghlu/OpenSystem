/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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
#ifndef _XT_CT_H
#define _XT_CT_H
#include <linux/types.h>
enum {
  XT_CT_NOTRACK = 1 << 0,
  XT_CT_NOTRACK_ALIAS = 1 << 1,
  XT_CT_ZONE_DIR_ORIG = 1 << 2,
  XT_CT_ZONE_DIR_REPL = 1 << 3,
  XT_CT_ZONE_MARK = 1 << 4,
  XT_CT_MASK = XT_CT_NOTRACK | XT_CT_NOTRACK_ALIAS | XT_CT_ZONE_DIR_ORIG | XT_CT_ZONE_DIR_REPL | XT_CT_ZONE_MARK,
};
struct xt_ct_target_info {
  __u16 flags;
  __u16 zone;
  __u32 ct_events;
  __u32 exp_events;
  char helper[16];
  struct nf_conn * ct __attribute__((aligned(8)));
};
struct xt_ct_target_info_v1 {
  __u16 flags;
  __u16 zone;
  __u32 ct_events;
  __u32 exp_events;
  char helper[16];
  char timeout[32];
  struct nf_conn * ct __attribute__((aligned(8)));
};
#endif
