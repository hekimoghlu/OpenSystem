/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
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
#ifndef _IP6T_AH_H
#define _IP6T_AH_H
#include <linux/types.h>
struct ip6t_ah {
  __u32 spis[2];
  __u32 hdrlen;
  __u8 hdrres;
  __u8 invflags;
};
#define IP6T_AH_SPI 0x01
#define IP6T_AH_LEN 0x02
#define IP6T_AH_RES 0x04
#define IP6T_AH_INV_SPI 0x01
#define IP6T_AH_INV_LEN 0x02
#define IP6T_AH_INV_MASK 0x03
#endif
