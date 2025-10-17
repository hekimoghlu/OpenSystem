/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#include "common_audio/signal_processing/include/signal_processing_library.h"

void WebRtcSpl_CrossCorrelation_mips(int32_t* cross_correlation,
                                     const int16_t* seq1,
                                     const int16_t* seq2,
                                     size_t dim_seq,
                                     size_t dim_cross_correlation,
                                     int right_shifts,
                                     int step_seq2) {

  int32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, sum = 0;
  int16_t *pseq2 = NULL;
  int16_t *pseq1 = NULL;
  int16_t *pseq1_0 = (int16_t*)&seq1[0];
  int16_t *pseq2_0 = (int16_t*)&seq2[0];
  int k = 0;

  __asm __volatile (
    ".set        push                                           \n\t"
    ".set        noreorder                                      \n\t"
    "sll         %[step_seq2], %[step_seq2],   1                \n\t"
    "andi        %[t0],        %[dim_seq],     1                \n\t"
    "bgtz        %[t0],        3f                               \n\t"
    " nop                                                       \n\t"
   "1:                                                          \n\t"
    "move        %[pseq1],     %[pseq1_0]                       \n\t"
    "move        %[pseq2],     %[pseq2_0]                       \n\t"
    "sra         %[k],         %[dim_seq],     1                \n\t"
    "addiu       %[dim_cc],    %[dim_cc],      -1               \n\t"
    "xor         %[sum],       %[sum],         %[sum]           \n\t"
   "2:                                                          \n\t"
    "lh          %[t0],        0(%[pseq1])                      \n\t"
    "lh          %[t1],        0(%[pseq2])                      \n\t"
    "lh          %[t2],        2(%[pseq1])                      \n\t"
    "lh          %[t3],        2(%[pseq2])                      \n\t"
    "mul         %[t0],        %[t0],          %[t1]            \n\t"
    "addiu       %[k],         %[k],           -1               \n\t"
    "mul         %[t2],        %[t2],          %[t3]            \n\t"
    "addiu       %[pseq1],     %[pseq1],       4                \n\t"
    "addiu       %[pseq2],     %[pseq2],       4                \n\t"
    "srav        %[t0],        %[t0],          %[right_shifts]  \n\t"
    "addu        %[sum],       %[sum],         %[t0]            \n\t"
    "srav        %[t2],        %[t2],          %[right_shifts]  \n\t"
    "bgtz        %[k],         2b                               \n\t"
    " addu       %[sum],       %[sum],         %[t2]            \n\t"
    "addu        %[pseq2_0],   %[pseq2_0],     %[step_seq2]     \n\t"
    "sw          %[sum],       0(%[cc])                         \n\t"
    "bgtz        %[dim_cc],    1b                               \n\t"
    " addiu      %[cc],        %[cc],          4                \n\t"
    "b           6f                                             \n\t"
    " nop                                                       \n\t"
   "3:                                                          \n\t"
    "move        %[pseq1],     %[pseq1_0]                       \n\t"
    "move        %[pseq2],     %[pseq2_0]                       \n\t"
    "sra         %[k],         %[dim_seq],     1                \n\t"
    "addiu       %[dim_cc],    %[dim_cc],      -1               \n\t"
    "beqz        %[k],         5f                               \n\t"
    " xor        %[sum],       %[sum],         %[sum]           \n\t"
   "4:                                                          \n\t"
    "lh          %[t0],        0(%[pseq1])                      \n\t"
    "lh          %[t1],        0(%[pseq2])                      \n\t"
    "lh          %[t2],        2(%[pseq1])                      \n\t"
    "lh          %[t3],        2(%[pseq2])                      \n\t"
    "mul         %[t0],        %[t0],          %[t1]            \n\t"
    "addiu       %[k],         %[k],           -1               \n\t"
    "mul         %[t2],        %[t2],          %[t3]            \n\t"
    "addiu       %[pseq1],     %[pseq1],       4                \n\t"
    "addiu       %[pseq2],     %[pseq2],       4                \n\t"
    "srav        %[t0],        %[t0],          %[right_shifts]  \n\t"
    "addu        %[sum],       %[sum],         %[t0]            \n\t"
    "srav        %[t2],        %[t2],          %[right_shifts]  \n\t"
    "bgtz        %[k],         4b                               \n\t"
    " addu       %[sum],       %[sum],         %[t2]            \n\t"
   "5:                                                          \n\t"
    "lh          %[t0],        0(%[pseq1])                      \n\t"
    "lh          %[t1],        0(%[pseq2])                      \n\t"
    "mul         %[t0],        %[t0],          %[t1]            \n\t"
    "srav        %[t0],        %[t0],          %[right_shifts]  \n\t"
    "addu        %[sum],       %[sum],         %[t0]            \n\t"
    "addu        %[pseq2_0],   %[pseq2_0],     %[step_seq2]     \n\t"
    "sw          %[sum],       0(%[cc])                         \n\t"
    "bgtz        %[dim_cc],    3b                               \n\t"
    " addiu      %[cc],        %[cc],          4                \n\t"
   "6:                                                          \n\t"
    ".set        pop                                            \n\t"
    : [step_seq2] "+r" (step_seq2), [t0] "=&r" (t0), [t1] "=&r" (t1),
      [t2] "=&r" (t2), [t3] "=&r" (t3), [pseq1] "=&r" (pseq1),
      [pseq2] "=&r" (pseq2), [pseq1_0] "+r" (pseq1_0), [pseq2_0] "+r" (pseq2_0),
      [k] "=&r" (k), [dim_cc] "+r" (dim_cross_correlation), [sum] "=&r" (sum),
      [cc] "+r" (cross_correlation)
    : [dim_seq] "r" (dim_seq), [right_shifts] "r" (right_shifts)
    : "hi", "lo", "memory"
  );
}
