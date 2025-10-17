/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
#ifndef _XT_MULTIPORT_H
#define _XT_MULTIPORT_H
#include <linux/types.h>
enum xt_multiport_flags {
  XT_MULTIPORT_SOURCE,
  XT_MULTIPORT_DESTINATION,
  XT_MULTIPORT_EITHER
};
#define XT_MULTI_PORTS 15
struct xt_multiport {
  __u8 flags;
  __u8 count;
  __u16 ports[XT_MULTI_PORTS];
};
struct xt_multiport_v1 {
  __u8 flags;
  __u8 count;
  __u16 ports[XT_MULTI_PORTS];
  __u8 pflags[XT_MULTI_PORTS];
  __u8 invert;
};
#endif
