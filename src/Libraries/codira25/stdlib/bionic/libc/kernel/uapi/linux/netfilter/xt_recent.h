/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
#ifndef _LINUX_NETFILTER_XT_RECENT_H
#define _LINUX_NETFILTER_XT_RECENT_H 1
#include <linux/types.h>
#include <linux/netfilter.h>
enum {
  XT_RECENT_CHECK = 1 << 0,
  XT_RECENT_SET = 1 << 1,
  XT_RECENT_UPDATE = 1 << 2,
  XT_RECENT_REMOVE = 1 << 3,
  XT_RECENT_TTL = 1 << 4,
  XT_RECENT_REAP = 1 << 5,
  XT_RECENT_SOURCE = 0,
  XT_RECENT_DEST = 1,
  XT_RECENT_NAME_LEN = 200,
};
#define XT_RECENT_MODIFIERS (XT_RECENT_TTL | XT_RECENT_REAP)
#define XT_RECENT_VALID_FLAGS (XT_RECENT_CHECK | XT_RECENT_SET | XT_RECENT_UPDATE | XT_RECENT_REMOVE | XT_RECENT_TTL | XT_RECENT_REAP)
struct xt_recent_mtinfo {
  __u32 seconds;
  __u32 hit_count;
  __u8 check_set;
  __u8 invert;
  char name[XT_RECENT_NAME_LEN];
  __u8 side;
};
struct xt_recent_mtinfo_v1 {
  __u32 seconds;
  __u32 hit_count;
  __u8 check_set;
  __u8 invert;
  char name[XT_RECENT_NAME_LEN];
  __u8 side;
  union nf_inet_addr mask;
};
#endif
