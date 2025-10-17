/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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
#include "./vp8_rtcd.h"
#include "vpx_ports/asmdefs_mmi.h"

#define COPY_MEM_16X2 \
  "gsldlc1    %[ftmp0],   0x07(%[src])                    \n\t" \
  "gsldrc1    %[ftmp0],   0x00(%[src])                    \n\t" \
  "ldl        %[tmp0],    0x0f(%[src])                    \n\t" \
  "ldr        %[tmp0],    0x08(%[src])                    \n\t" \
  MMI_ADDU(%[src],     %[src],         %[src_stride])           \
  "gssdlc1    %[ftmp0],   0x07(%[dst])                    \n\t" \
  "gssdrc1    %[ftmp0],   0x00(%[dst])                    \n\t" \
  "sdl        %[tmp0],    0x0f(%[dst])                    \n\t" \
  "sdr        %[tmp0],    0x08(%[dst])                    \n\t" \
  MMI_ADDU(%[dst],      %[dst],        %[dst_stride])           \
  "gsldlc1    %[ftmp1],   0x07(%[src])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src])                    \n\t" \
  "ldl        %[tmp1],    0x0f(%[src])                    \n\t" \
  "ldr        %[tmp1],    0x08(%[src])                    \n\t" \
  MMI_ADDU(%[src],     %[src],         %[src_stride])           \
  "gssdlc1    %[ftmp1],   0x07(%[dst])                    \n\t" \
  "gssdrc1    %[ftmp1],   0x00(%[dst])                    \n\t" \
  "sdl        %[tmp1],    0x0f(%[dst])                    \n\t" \
  "sdr        %[tmp1],    0x08(%[dst])                    \n\t" \
  MMI_ADDU(%[dst],     %[dst],         %[dst_stride])

#define COPY_MEM_8X2 \
  "gsldlc1    %[ftmp0],   0x07(%[src])                    \n\t" \
  "gsldrc1    %[ftmp0],   0x00(%[src])                    \n\t" \
  MMI_ADDU(%[src],     %[src],         %[src_stride])           \
  "ldl        %[tmp0],    0x07(%[src])                    \n\t" \
  "ldr        %[tmp0],    0x00(%[src])                    \n\t" \
  MMI_ADDU(%[src],     %[src],         %[src_stride])           \
                                                                \
  "gssdlc1    %[ftmp0],   0x07(%[dst])                    \n\t" \
  "gssdrc1    %[ftmp0],   0x00(%[dst])                    \n\t" \
  MMI_ADDU(%[dst],      %[dst],        %[dst_stride])           \
  "sdl        %[tmp0],    0x07(%[dst])                    \n\t" \
  "sdr        %[tmp0],    0x00(%[dst])                    \n\t" \
  MMI_ADDU(%[dst],     %[dst],         %[dst_stride])

void vp8_copy_mem16x16_mmi(unsigned char *src, int src_stride,
                           unsigned char *dst, int dst_stride) {
  double ftmp[2];
  uint64_t tmp[2];
  uint8_t loop_count = 4;

  /* clang-format off */
  __asm__ volatile (
    "1:                                                     \n\t"
    COPY_MEM_16X2
    COPY_MEM_16X2
    MMI_ADDIU(%[loop_count], %[loop_count], -0x01)
    "bnez       %[loop_count],    1b                        \n\t"
    : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
      [tmp0]"=&r"(tmp[0]),              [tmp1]"=&r"(tmp[1]),
      [loop_count]"+&r"(loop_count),
      [dst]"+&r"(dst),                  [src]"+&r"(src)
    : [src_stride]"r"((mips_reg)src_stride),
      [dst_stride]"r"((mips_reg)dst_stride)
    : "memory"
  );
  /* clang-format on */
}

void vp8_copy_mem8x8_mmi(unsigned char *src, int src_stride, unsigned char *dst,
                         int dst_stride) {
  double ftmp[2];
  uint64_t tmp[1];
  uint8_t loop_count = 4;

  /* clang-format off */
  __asm__ volatile (
    "1:                                                     \n\t"
    COPY_MEM_8X2
    MMI_ADDIU(%[loop_count], %[loop_count], -0x01)
    "bnez       %[loop_count],    1b                        \n\t"
    : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
      [tmp0]"=&r"(tmp[0]),              [loop_count]"+&r"(loop_count),
      [dst]"+&r"(dst),                  [src]"+&r"(src)
    : [src_stride]"r"((mips_reg)src_stride),
      [dst_stride]"r"((mips_reg)dst_stride)
    : "memory"
  );
  /* clang-format on */
}

void vp8_copy_mem8x4_mmi(unsigned char *src, int src_stride, unsigned char *dst,
                         int dst_stride) {
  double ftmp[2];
  uint64_t tmp[1];

  /* clang-format off */
  __asm__ volatile (
    COPY_MEM_8X2
    COPY_MEM_8X2
    : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
      [tmp0]"=&r"(tmp[0]),
      [dst]"+&r"(dst),                  [src]"+&r"(src)
    : [src_stride]"r"((mips_reg)src_stride),
      [dst_stride]"r"((mips_reg)dst_stride)
    : "memory"
  );
  /* clang-format on */
}
