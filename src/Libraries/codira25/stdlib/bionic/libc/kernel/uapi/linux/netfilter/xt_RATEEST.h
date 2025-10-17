/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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
#ifndef _XT_RATEEST_MATCH_H
#define _XT_RATEEST_MATCH_H
#include <linux/types.h>
#include <linux/if.h>
enum xt_rateest_match_flags {
  XT_RATEEST_MATCH_INVERT = 1 << 0,
  XT_RATEEST_MATCH_ABS = 1 << 1,
  XT_RATEEST_MATCH_REL = 1 << 2,
  XT_RATEEST_MATCH_DELTA = 1 << 3,
  XT_RATEEST_MATCH_BPS = 1 << 4,
  XT_RATEEST_MATCH_PPS = 1 << 5,
};
enum xt_rateest_match_mode {
  XT_RATEEST_MATCH_NONE,
  XT_RATEEST_MATCH_EQ,
  XT_RATEEST_MATCH_LT,
  XT_RATEEST_MATCH_GT,
};
struct xt_rateest_match_info {
  char name1[IFNAMSIZ];
  char name2[IFNAMSIZ];
  __u16 flags;
  __u16 mode;
  __u32 bps1;
  __u32 pps1;
  __u32 bps2;
  __u32 pps2;
  struct xt_rateest * est1 __attribute__((aligned(8)));
  struct xt_rateest * est2 __attribute__((aligned(8)));
};
#endif
