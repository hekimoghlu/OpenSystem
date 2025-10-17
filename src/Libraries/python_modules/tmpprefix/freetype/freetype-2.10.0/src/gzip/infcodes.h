/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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

#ifndef _INFCODES_H
#define _INFCODES_H

struct inflate_codes_state;
typedef struct inflate_codes_state FAR inflate_codes_statef;

local inflate_codes_statef *inflate_codes_new OF((
    uInt, uInt,
    inflate_huft *, inflate_huft *,
    z_streamp ));

local int inflate_codes OF((
    inflate_blocks_statef *,
    z_streamp ,
    int));

local void inflate_codes_free OF((
    inflate_codes_statef *,
    z_streamp ));

#endif /* _INFCODES_H */
