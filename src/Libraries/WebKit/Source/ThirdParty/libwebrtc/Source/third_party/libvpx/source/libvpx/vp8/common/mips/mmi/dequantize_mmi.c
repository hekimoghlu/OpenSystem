/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
#include "vp8/common/blockd.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/asmdefs_mmi.h"

void vp8_dequantize_b_mmi(BLOCKD *d, int16_t *DQC) {
  double ftmp[8];

  __asm__ volatile(
      "gsldlc1    %[ftmp0],   0x07(%[qcoeff])                 \n\t"
      "gsldrc1    %[ftmp0],   0x00(%[qcoeff])                 \n\t"
      "gsldlc1    %[ftmp1],   0x0f(%[qcoeff])                 \n\t"
      "gsldrc1    %[ftmp1],   0x08(%[qcoeff])                 \n\t"
      "gsldlc1    %[ftmp2],   0x17(%[qcoeff])                 \n\t"
      "gsldrc1    %[ftmp2],   0x10(%[qcoeff])                 \n\t"
      "gsldlc1    %[ftmp3],   0x1f(%[qcoeff])                 \n\t"
      "gsldrc1    %[ftmp3],   0x18(%[qcoeff])                 \n\t"

      "gsldlc1    %[ftmp4],   0x07(%[DQC])                    \n\t"
      "gsldrc1    %[ftmp4],   0x00(%[DQC])                    \n\t"
      "gsldlc1    %[ftmp5],   0x0f(%[DQC])                    \n\t"
      "gsldrc1    %[ftmp5],   0x08(%[DQC])                    \n\t"
      "gsldlc1    %[ftmp6],   0x17(%[DQC])                    \n\t"
      "gsldrc1    %[ftmp6],   0x10(%[DQC])                    \n\t"
      "gsldlc1    %[ftmp7],   0x1f(%[DQC])                    \n\t"
      "gsldrc1    %[ftmp7],   0x18(%[DQC])                    \n\t"

      "pmullh     %[ftmp0],   %[ftmp0],       %[ftmp4]        \n\t"
      "pmullh     %[ftmp1],   %[ftmp1],       %[ftmp5]        \n\t"
      "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp6]        \n\t"
      "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp7]        \n\t"

      "gssdlc1    %[ftmp0],   0x07(%[dqcoeff])                \n\t"
      "gssdrc1    %[ftmp0],   0x00(%[dqcoeff])                \n\t"
      "gssdlc1    %[ftmp1],   0x0f(%[dqcoeff])                \n\t"
      "gssdrc1    %[ftmp1],   0x08(%[dqcoeff])                \n\t"
      "gssdlc1    %[ftmp2],   0x17(%[dqcoeff])                \n\t"
      "gssdrc1    %[ftmp2],   0x10(%[dqcoeff])                \n\t"
      "gssdlc1    %[ftmp3],   0x1f(%[dqcoeff])                \n\t"
      "gssdrc1    %[ftmp3],   0x18(%[dqcoeff])                \n\t"
      : [ftmp0] "=&f"(ftmp[0]), [ftmp1] "=&f"(ftmp[1]), [ftmp2] "=&f"(ftmp[2]),
        [ftmp3] "=&f"(ftmp[3]), [ftmp4] "=&f"(ftmp[4]), [ftmp5] "=&f"(ftmp[5]),
        [ftmp6] "=&f"(ftmp[6]), [ftmp7] "=&f"(ftmp[7])
      : [dqcoeff] "r"(d->dqcoeff), [qcoeff] "r"(d->qcoeff), [DQC] "r"(DQC)
      : "memory");
}

void vp8_dequant_idct_add_mmi(int16_t *input, int16_t *dq, unsigned char *dest,
                              int stride) {
  double ftmp[8];

  __asm__ volatile(
      "gsldlc1    %[ftmp0],   0x07(%[dq])                     \n\t"
      "gsldrc1    %[ftmp0],   0x00(%[dq])                     \n\t"
      "gsldlc1    %[ftmp1],   0x0f(%[dq])                     \n\t"
      "gsldrc1    %[ftmp1],   0x08(%[dq])                     \n\t"
      "gsldlc1    %[ftmp2],   0x17(%[dq])                     \n\t"
      "gsldrc1    %[ftmp2],   0x10(%[dq])                     \n\t"
      "gsldlc1    %[ftmp3],   0x1f(%[dq])                     \n\t"
      "gsldrc1    %[ftmp3],   0x18(%[dq])                     \n\t"

      "gsldlc1    %[ftmp4],   0x07(%[input])                  \n\t"
      "gsldrc1    %[ftmp4],   0x00(%[input])                  \n\t"
      "gsldlc1    %[ftmp5],   0x0f(%[input])                  \n\t"
      "gsldrc1    %[ftmp5],   0x08(%[input])                  \n\t"
      "gsldlc1    %[ftmp6],   0x17(%[input])                  \n\t"
      "gsldrc1    %[ftmp6],   0x10(%[input])                  \n\t"
      "gsldlc1    %[ftmp7],   0x1f(%[input])                  \n\t"
      "gsldrc1    %[ftmp7],   0x18(%[input])                  \n\t"

      "pmullh     %[ftmp0],   %[ftmp0],       %[ftmp4]        \n\t"
      "pmullh     %[ftmp1],   %[ftmp1],       %[ftmp5]        \n\t"
      "pmullh     %[ftmp2],   %[ftmp2],       %[ftmp6]        \n\t"
      "pmullh     %[ftmp3],   %[ftmp3],       %[ftmp7]        \n\t"

      "gssdlc1    %[ftmp0],   0x07(%[input])                  \n\t"
      "gssdrc1    %[ftmp0],   0x00(%[input])                  \n\t"
      "gssdlc1    %[ftmp1],   0x0f(%[input])                  \n\t"
      "gssdrc1    %[ftmp1],   0x08(%[input])                  \n\t"
      "gssdlc1    %[ftmp2],   0x17(%[input])                  \n\t"
      "gssdrc1    %[ftmp2],   0x10(%[input])                  \n\t"
      "gssdlc1    %[ftmp3],   0x1f(%[input])                  \n\t"
      "gssdrc1    %[ftmp3],   0x18(%[input])                  \n\t"
      : [ftmp0] "=&f"(ftmp[0]), [ftmp1] "=&f"(ftmp[1]), [ftmp2] "=&f"(ftmp[2]),
        [ftmp3] "=&f"(ftmp[3]), [ftmp4] "=&f"(ftmp[4]), [ftmp5] "=&f"(ftmp[5]),
        [ftmp6] "=&f"(ftmp[6]), [ftmp7] "=&f"(ftmp[7])
      : [dq] "r"(dq), [input] "r"(input)
      : "memory");

  vp8_short_idct4x4llm_mmi(input, dest, stride, dest, stride);

  __asm__ volatile(
      "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]        \n\t"
      "gssdlc1    %[ftmp0],   0x07(%[input])                  \n\t"
      "gssdrc1    %[ftmp0],   0x00(%[input])                  \n\t"
      "sdl        $0,         0x0f(%[input])                  \n\t"
      "sdr        $0,         0x08(%[input])                  \n\t"
      "gssdlc1    %[ftmp0],   0x17(%[input])                  \n\t"
      "gssdrc1    %[ftmp0],   0x10(%[input])                  \n\t"
      "sdl        $0,         0x1f(%[input])                  \n\t"
      "sdr        $0,         0x18(%[input])                  \n\t"
      : [ftmp0] "=&f"(ftmp[0])
      : [input] "r"(input)
      : "memory");
}
