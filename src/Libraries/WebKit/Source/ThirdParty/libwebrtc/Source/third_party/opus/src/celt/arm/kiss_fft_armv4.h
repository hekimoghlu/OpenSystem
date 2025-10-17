/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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
#ifndef KISS_FFT_ARMv4_H
#define KISS_FFT_ARMv4_H

#if !defined(KISS_FFT_GUTS_H)
#error "This file should only be included from _kiss_fft_guts.h"
#endif

#ifdef FIXED_POINT

#undef C_MUL
#define C_MUL(m,a,b) \
    do{ \
       int br__; \
       int bi__; \
       int tt__; \
        __asm__ __volatile__( \
            "#C_MUL\n\t" \
            "ldrsh %[br], [%[bp], #0]\n\t" \
            "ldm %[ap], {r0,r1}\n\t" \
            "ldrsh %[bi], [%[bp], #2]\n\t" \
            "smull %[tt], %[mi], r1, %[br]\n\t" \
            "smlal %[tt], %[mi], r0, %[bi]\n\t" \
            "rsb %[bi], %[bi], #0\n\t" \
            "smull %[br], %[mr], r0, %[br]\n\t" \
            "mov %[tt], %[tt], lsr #15\n\t" \
            "smlal %[br], %[mr], r1, %[bi]\n\t" \
            "orr %[mi], %[tt], %[mi], lsl #17\n\t" \
            "mov %[br], %[br], lsr #15\n\t" \
            "orr %[mr], %[br], %[mr], lsl #17\n\t" \
            : [mr]"=r"((m).r), [mi]"=r"((m).i), \
              [br]"=&r"(br__), [bi]"=r"(bi__), [tt]"=r"(tt__) \
            : [ap]"r"(&(a)), [bp]"r"(&(b)) \
            : "r0", "r1" \
        ); \
    } \
    while(0)

#undef C_MUL4
#define C_MUL4(m,a,b) \
    do{ \
       int br__; \
       int bi__; \
       int tt__; \
        __asm__ __volatile__( \
            "#C_MUL4\n\t" \
            "ldrsh %[br], [%[bp], #0]\n\t" \
            "ldm %[ap], {r0,r1}\n\t" \
            "ldrsh %[bi], [%[bp], #2]\n\t" \
            "smull %[tt], %[mi], r1, %[br]\n\t" \
            "smlal %[tt], %[mi], r0, %[bi]\n\t" \
            "rsb %[bi], %[bi], #0\n\t" \
            "smull %[br], %[mr], r0, %[br]\n\t" \
            "mov %[tt], %[tt], lsr #17\n\t" \
            "smlal %[br], %[mr], r1, %[bi]\n\t" \
            "orr %[mi], %[tt], %[mi], lsl #15\n\t" \
            "mov %[br], %[br], lsr #17\n\t" \
            "orr %[mr], %[br], %[mr], lsl #15\n\t" \
            : [mr]"=r"((m).r), [mi]"=r"((m).i), \
              [br]"=&r"(br__), [bi]"=r"(bi__), [tt]"=r"(tt__) \
            : [ap]"r"(&(a)), [bp]"r"(&(b)) \
            : "r0", "r1" \
        ); \
    } \
    while(0)

#undef C_MULC
#define C_MULC(m,a,b) \
    do{ \
       int br__; \
       int bi__; \
       int tt__; \
        __asm__ __volatile__( \
            "#C_MULC\n\t" \
            "ldrsh %[br], [%[bp], #0]\n\t" \
            "ldm %[ap], {r0,r1}\n\t" \
            "ldrsh %[bi], [%[bp], #2]\n\t" \
            "smull %[tt], %[mr], r0, %[br]\n\t" \
            "smlal %[tt], %[mr], r1, %[bi]\n\t" \
            "rsb %[bi], %[bi], #0\n\t" \
            "smull %[br], %[mi], r1, %[br]\n\t" \
            "mov %[tt], %[tt], lsr #15\n\t" \
            "smlal %[br], %[mi], r0, %[bi]\n\t" \
            "orr %[mr], %[tt], %[mr], lsl #17\n\t" \
            "mov %[br], %[br], lsr #15\n\t" \
            "orr %[mi], %[br], %[mi], lsl #17\n\t" \
            : [mr]"=r"((m).r), [mi]"=r"((m).i), \
              [br]"=&r"(br__), [bi]"=r"(bi__), [tt]"=r"(tt__) \
            : [ap]"r"(&(a)), [bp]"r"(&(b)) \
            : "r0", "r1" \
        ); \
    } \
    while(0)

#endif /* FIXED_POINT */

#endif /* KISS_FFT_ARMv4_H */
