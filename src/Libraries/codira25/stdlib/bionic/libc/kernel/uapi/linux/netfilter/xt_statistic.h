/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#ifndef _XT_STATISTIC_H
#define _XT_STATISTIC_H
#include <linux/types.h>
enum xt_statistic_mode {
  XT_STATISTIC_MODE_RANDOM,
  XT_STATISTIC_MODE_NTH,
  __XT_STATISTIC_MODE_MAX
};
#define XT_STATISTIC_MODE_MAX (__XT_STATISTIC_MODE_MAX - 1)
enum xt_statistic_flags {
  XT_STATISTIC_INVERT = 0x1,
};
#define XT_STATISTIC_MASK 0x1
struct xt_statistic_priv;
struct xt_statistic_info {
  __u16 mode;
  __u16 flags;
  union {
    struct {
      __u32 probability;
    } random;
    struct {
      __u32 every;
      __u32 packet;
      __u32 count;
    } nth;
  } u;
  struct xt_statistic_priv * master __attribute__((aligned(8)));
};
#endif
