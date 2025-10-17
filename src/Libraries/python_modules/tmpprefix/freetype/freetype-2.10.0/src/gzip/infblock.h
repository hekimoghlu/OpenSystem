/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
/* WARNING: this file should *not* be used by applications. It is
   part of the implementation of the compression library and is
   subject to change. Applications should only use zlib.h.
 */

#ifndef _INFBLOCK_H
#define _INFBLOCK_H

struct inflate_blocks_state;
typedef struct inflate_blocks_state FAR inflate_blocks_statef;

local  inflate_blocks_statef * inflate_blocks_new OF((
    z_streamp z,
    check_func c,               /* check function */
    uInt w));                   /* window size */

local  int inflate_blocks OF((
    inflate_blocks_statef *,
    z_streamp ,
    int));                      /* initial return code */

local  void inflate_blocks_reset OF((
    inflate_blocks_statef *,
    z_streamp ,
    uLongf *));                  /* check value on output */

local  int inflate_blocks_free OF((
    inflate_blocks_statef *,
    z_streamp));

#endif /* _INFBLOCK_H */
