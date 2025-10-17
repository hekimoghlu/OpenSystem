/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#ifndef _XT_OWNER_MATCH_H
#define _XT_OWNER_MATCH_H
#include <linux/types.h>
enum {
  XT_OWNER_UID = 1 << 0,
  XT_OWNER_GID = 1 << 1,
  XT_OWNER_SOCKET = 1 << 2,
  XT_OWNER_SUPPL_GROUPS = 1 << 3,
};
#define XT_OWNER_MASK (XT_OWNER_UID | XT_OWNER_GID | XT_OWNER_SOCKET | XT_OWNER_SUPPL_GROUPS)
struct xt_owner_match_info {
  __u32 uid_min, uid_max;
  __u32 gid_min, gid_max;
  __u8 match, invert;
};
#endif
