/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
#ifndef __LINUX_BRIDGE_EBT_MARK_T_H
#define __LINUX_BRIDGE_EBT_MARK_T_H
#define MARK_SET_VALUE (0xfffffff0)
#define MARK_OR_VALUE (0xffffffe0)
#define MARK_AND_VALUE (0xffffffd0)
#define MARK_XOR_VALUE (0xffffffc0)
struct ebt_mark_t_info {
  unsigned long mark;
  int target;
};
#define EBT_MARK_TARGET "mark"
#endif
