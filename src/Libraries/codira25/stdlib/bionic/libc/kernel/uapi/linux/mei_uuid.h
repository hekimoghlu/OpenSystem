/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
#ifndef _UAPI_LINUX_MEI_UUID_H_
#define _UAPI_LINUX_MEI_UUID_H_
#include <linux/types.h>
typedef struct {
  __u8 b[16];
} uuid_le;
#define UUID_LE(a,b,c,d0,d1,d2,d3,d4,d5,d6,d7) \
((uuid_le) \
{ { (a) & 0xff, ((a) >> 8) & 0xff, ((a) >> 16) & 0xff, ((a) >> 24) & 0xff, (b) & 0xff, ((b) >> 8) & 0xff, (c) & 0xff, ((c) >> 8) & 0xff, (d0), (d1), (d2), (d3), (d4), (d5), (d6), (d7) } })
#define NULL_UUID_LE UUID_LE(0x00000000, 0x0000, 0x0000, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
#endif
