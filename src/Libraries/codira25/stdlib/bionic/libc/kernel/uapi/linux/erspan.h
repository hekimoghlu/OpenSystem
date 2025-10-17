/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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
#ifndef _UAPI_ERSPAN_H
#define _UAPI_ERSPAN_H
#include <linux/types.h>
#include <asm/byteorder.h>
struct erspan_md2 {
  __be32 timestamp;
  __be16 sgt;
#ifdef __LITTLE_ENDIAN_BITFIELD
  __u8 hwid_upper : 2, ft : 5, p : 1;
  __u8 o : 1, gra : 2, dir : 1, hwid : 4;
#elif defined(__BIG_ENDIAN_BITFIELD)
  __u8 p : 1, ft : 5, hwid_upper : 2;
  __u8 hwid : 4, dir : 1, gra : 2, o : 1;
#else
#error "Please fix <asm/byteorder.h>"
#endif
};
struct erspan_metadata {
  int version;
  union {
    __be32 index;
    struct erspan_md2 md2;
  } u;
};
#endif
