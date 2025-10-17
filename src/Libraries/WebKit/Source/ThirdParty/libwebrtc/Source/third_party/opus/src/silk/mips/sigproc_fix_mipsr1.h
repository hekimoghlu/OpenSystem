/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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
#ifndef SILK_SIGPROC_FIX_MIPSR1_H
#define SILK_SIGPROC_FIX_MIPSR1_H

#undef silk_SAT16
static inline short int silk_SAT16(int a)
{
    int c;
    c = __builtin_mips_shll_s_w(a, 16);
    c = c>>16;

    return c;
}

#undef silk_LSHIFT_SAT32
static inline int silk_LSHIFT_SAT32(int a, int shift)
{
    int r;

    r = __builtin_mips_shll_s_w(a, shift);

    return r;
}

#undef silk_RSHIFT_ROUND
static inline int silk_RSHIFT_ROUND(int a, int shift)
{
    int r;

    r = __builtin_mips_shra_r_w(a, shift);
    return r;
}

#endif /* SILK_SIGPROC_FIX_MIPSR1_H */
