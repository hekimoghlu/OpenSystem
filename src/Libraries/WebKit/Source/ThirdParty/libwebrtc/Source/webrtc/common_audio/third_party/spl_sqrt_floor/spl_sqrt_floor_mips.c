/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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
// Minor modifications in code style for WebRTC, 2012.
// Code optimizations for MIPS, 2013.

#include "common_audio/third_party/spl_sqrt_floor/spl_sqrt_floor.h"

/*
 * Algorithm:
 * Successive approximation of the equation (root + delta) ^ 2 = N
 * until delta < 1. If delta < 1 we have the integer part of SQRT (N).
 * Use delta = 2^i for i = 15 .. 0.
 *
 * Output precision is 16 bits. Note for large input values (close to
 * 0x7FFFFFFF), bit 15 (the highest bit of the low 16-bit half word)
 * contains the MSB information (a non-sign value). Do with caution
 * if you need to cast the output to int16_t type.
 *
 * If the input value is negative, it returns 0.
 */


int32_t WebRtcSpl_SqrtFloor(int32_t value)
{
  int32_t root = 0, tmp1, tmp2, tmp3, tmp4;

  __asm __volatile(
    ".set   push                                       \n\t"
    ".set   noreorder                                  \n\t"

    "lui    %[tmp1],      0x4000                       \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "sub    %[tmp3],      %[value],     %[tmp1]        \n\t"
    "lui    %[tmp1],      0x1                          \n\t"
    "or     %[tmp4],      %[root],      %[tmp1]        \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x4000         \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      14                           \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x8000         \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x2000         \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      13                           \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x4000         \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x1000         \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      12                           \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x2000         \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x800          \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      11                           \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x1000         \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x400          \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      10                           \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x800          \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x200          \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      9                            \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],       0x400         \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x100          \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      8                            \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x200          \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x80           \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      7                            \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x100          \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x40           \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      6                            \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x80           \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x20           \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      5                            \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x40           \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x10           \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      4                            \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x20           \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x8            \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      3                            \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x10           \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x4            \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      2                            \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x8            \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x2            \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "sll    %[tmp1],      1                            \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "subu   %[tmp3],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x4            \n\t"
    "movz   %[value],     %[tmp3],      %[tmp2]        \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    "addiu  %[tmp1],      $0,           0x1            \n\t"
    "addu   %[tmp1],      %[tmp1],      %[root]        \n\t"
    "slt    %[tmp2],      %[value],     %[tmp1]        \n\t"
    "ori    %[tmp4],      %[root],      0x2            \n\t"
    "movz   %[root],      %[tmp4],      %[tmp2]        \n\t"

    ".set   pop                                        \n\t"

    : [root] "+r" (root), [value] "+r" (value),
      [tmp1] "=&r" (tmp1), [tmp2] "=&r" (tmp2),
      [tmp3] "=&r" (tmp3), [tmp4] "=&r" (tmp4)
    :
  );

  return root >> 1;
}

