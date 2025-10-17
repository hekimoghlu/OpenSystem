/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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
#ifndef _XT_NFLOG_TARGET
#define _XT_NFLOG_TARGET
#include <linux/types.h>
#define XT_NFLOG_DEFAULT_GROUP 0x1
#define XT_NFLOG_DEFAULT_THRESHOLD 0
#define XT_NFLOG_MASK 0x1
#define XT_NFLOG_F_COPY_LEN 0x1
struct xt_nflog_info {
  __u32 len;
  __u16 group;
  __u16 threshold;
  __u16 flags;
  __u16 pad;
  char prefix[64];
};
#endif
