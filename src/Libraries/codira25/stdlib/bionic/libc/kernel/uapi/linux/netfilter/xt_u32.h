/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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
#ifndef _XT_U32_H
#define _XT_U32_H 1
#include <linux/types.h>
enum xt_u32_ops {
  XT_U32_AND,
  XT_U32_LEFTSH,
  XT_U32_RIGHTSH,
  XT_U32_AT,
};
struct xt_u32_location_element {
  __u32 number;
  __u8 nextop;
};
struct xt_u32_value_element {
  __u32 min;
  __u32 max;
};
#define XT_U32_MAXSIZE 10
struct xt_u32_test {
  struct xt_u32_location_element location[XT_U32_MAXSIZE + 1];
  struct xt_u32_value_element value[XT_U32_MAXSIZE + 1];
  __u8 nnums;
  __u8 nvalues;
};
struct xt_u32 {
  struct xt_u32_test tests[XT_U32_MAXSIZE + 1];
  __u8 ntests;
  __u8 invert;
};
#endif
