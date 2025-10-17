/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#ifndef _XT_SECMARK_H_target
#define _XT_SECMARK_H_target
#include <linux/types.h>
#define SECMARK_MODE_SEL 0x01
#define SECMARK_SECCTX_MAX 256
struct xt_secmark_target_info {
  __u8 mode;
  __u32 secid;
  char secctx[SECMARK_SECCTX_MAX];
};
struct xt_secmark_target_info_v1 {
  __u8 mode;
  char secctx[SECMARK_SECCTX_MAX];
  __u32 secid;
};
#endif
