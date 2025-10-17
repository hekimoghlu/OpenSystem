/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#ifndef __LINUX_IF_ADDRLABEL_H
#define __LINUX_IF_ADDRLABEL_H
#include <linux/types.h>
struct ifaddrlblmsg {
  __u8 ifal_family;
  __u8 __ifal_reserved;
  __u8 ifal_prefixlen;
  __u8 ifal_flags;
  __u32 ifal_index;
  __u32 ifal_seq;
};
enum {
  IFAL_ADDRESS = 1,
  IFAL_LABEL = 2,
  __IFAL_MAX
};
#define IFAL_MAX (__IFAL_MAX - 1)
#endif
