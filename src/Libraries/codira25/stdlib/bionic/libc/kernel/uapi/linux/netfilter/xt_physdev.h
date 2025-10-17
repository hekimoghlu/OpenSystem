/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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
#ifndef _UAPI_XT_PHYSDEV_H
#define _UAPI_XT_PHYSDEV_H
#include <linux/types.h>
#include <linux/if.h>
#define XT_PHYSDEV_OP_IN 0x01
#define XT_PHYSDEV_OP_OUT 0x02
#define XT_PHYSDEV_OP_BRIDGED 0x04
#define XT_PHYSDEV_OP_ISIN 0x08
#define XT_PHYSDEV_OP_ISOUT 0x10
#define XT_PHYSDEV_OP_MASK (0x20 - 1)
struct xt_physdev_info {
  char physindev[IFNAMSIZ];
  char in_mask[IFNAMSIZ];
  char physoutdev[IFNAMSIZ];
  char out_mask[IFNAMSIZ];
  __u8 invert;
  __u8 bitmask;
};
#endif
