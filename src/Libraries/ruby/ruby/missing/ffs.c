/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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
/* ffs() is defined by Single Unix Specification. */

#include "ruby.h"

int ffs(int arg)
{
    unsigned int x = (unsigned int)arg;
    int r;

    if (x == 0)
        return 0;

    r = 1;

#if 32 < SIZEOF_INT * CHAR_BIT
    if ((x & 0xffffffff) == 0) {
        x >>= 32;
        r += 32;
    }
#endif

    if ((x & 0xffff) == 0) {
        x >>= 16;
        r += 16;
    }

    if ((x & 0xff) == 0) {
        x >>= 8;
        r += 8;
    }

    if ((x & 0xf) == 0) {
        x >>= 4;
        r += 4;
    }

    if ((x & 0x3) == 0) {
        x >>= 2;
        r += 2;
    }

    if ((x & 0x1) == 0) {
        x >>= 1;
        r += 1;
    }

    return r;
}
