/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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
#ifndef _LINUX_NETFILTER_XT_L2TP_H
#define _LINUX_NETFILTER_XT_L2TP_H
#include <linux/types.h>
enum xt_l2tp_type {
  XT_L2TP_TYPE_CONTROL,
  XT_L2TP_TYPE_DATA,
};
struct xt_l2tp_info {
  __u32 tid;
  __u32 sid;
  __u8 version;
  __u8 type;
  __u8 flags;
};
enum {
  XT_L2TP_TID = (1 << 0),
  XT_L2TP_SID = (1 << 1),
  XT_L2TP_VERSION = (1 << 2),
  XT_L2TP_TYPE = (1 << 3),
};
#endif
