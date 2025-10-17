/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#ifndef _UAPI_LINUX_HW_BREAKPOINT_H
#define _UAPI_LINUX_HW_BREAKPOINT_H
enum {
  HW_BREAKPOINT_LEN_1 = 1,
  HW_BREAKPOINT_LEN_2 = 2,
  HW_BREAKPOINT_LEN_3 = 3,
  HW_BREAKPOINT_LEN_4 = 4,
  HW_BREAKPOINT_LEN_5 = 5,
  HW_BREAKPOINT_LEN_6 = 6,
  HW_BREAKPOINT_LEN_7 = 7,
  HW_BREAKPOINT_LEN_8 = 8,
};
enum {
  HW_BREAKPOINT_EMPTY = 0,
  HW_BREAKPOINT_R = 1,
  HW_BREAKPOINT_W = 2,
  HW_BREAKPOINT_RW = HW_BREAKPOINT_R | HW_BREAKPOINT_W,
  HW_BREAKPOINT_X = 4,
  HW_BREAKPOINT_INVALID = HW_BREAKPOINT_RW | HW_BREAKPOINT_X,
};
#endif
