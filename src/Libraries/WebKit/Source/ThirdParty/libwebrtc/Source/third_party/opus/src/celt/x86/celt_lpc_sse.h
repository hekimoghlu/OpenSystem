/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
#ifndef CELT_LPC_SSE_H
#define CELT_LPC_SSE_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if defined(OPUS_X86_MAY_HAVE_SSE4_1) && defined(FIXED_POINT)

void celt_fir_sse4_1(
         const opus_val16 *x,
         const opus_val16 *num,
         opus_val16 *y,
         int N,
         int ord,
         int arch);

#if defined(OPUS_X86_PRESUME_SSE4_1)
#define OVERRIDE_CELT_FIR
#define celt_fir(x, num, y, N, ord, arch) \
    ((void)arch, celt_fir_sse4_1(x, num, y, N, ord, arch))

#elif defined(OPUS_HAVE_RTCD)

extern void (*const CELT_FIR_IMPL[OPUS_ARCHMASK + 1])(
         const opus_val16 *x,
         const opus_val16 *num,
         opus_val16 *y,
         int N,
         int ord,
         int arch);

#define OVERRIDE_CELT_FIR
#  define celt_fir(x, num, y, N, ord, arch) \
    ((*CELT_FIR_IMPL[(arch) & OPUS_ARCHMASK])(x, num, y, N, ord, arch))

#endif
#endif

#endif
