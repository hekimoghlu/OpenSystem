/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
#include <stdio.h>
#include "blockd.h"

void vp8_print_modes_and_motion_vectors(MODE_INFO *mi, int rows, int cols,
                                        int frame) {
  int mb_row;
  int mb_col;
  int mb_index = 0;
  FILE *mvs = fopen("mvs.stt", "a");

  /* print out the macroblock Y modes */
  mb_index = 0;
  fprintf(mvs, "Mb Modes for Frame %d\n", frame);

  for (mb_row = 0; mb_row < rows; ++mb_row) {
    for (mb_col = 0; mb_col < cols; ++mb_col) {
      fprintf(mvs, "%2d ", mi[mb_index].mbmi.mode);

      mb_index++;
    }

    fprintf(mvs, "\n");
    mb_index++;
  }

  fprintf(mvs, "\n");

  mb_index = 0;
  fprintf(mvs, "Mb mv ref for Frame %d\n", frame);

  for (mb_row = 0; mb_row < rows; ++mb_row) {
    for (mb_col = 0; mb_col < cols; ++mb_col) {
      fprintf(mvs, "%2d ", mi[mb_index].mbmi.ref_frame);

      mb_index++;
    }

    fprintf(mvs, "\n");
    mb_index++;
  }

  fprintf(mvs, "\n");

  /* print out the macroblock UV modes */
  mb_index = 0;
  fprintf(mvs, "UV Modes for Frame %d\n", frame);

  for (mb_row = 0; mb_row < rows; ++mb_row) {
    for (mb_col = 0; mb_col < cols; ++mb_col) {
      fprintf(mvs, "%2d ", mi[mb_index].mbmi.uv_mode);

      mb_index++;
    }

    mb_index++;
    fprintf(mvs, "\n");
  }

  fprintf(mvs, "\n");

  /* print out the block modes */
  fprintf(mvs, "Mbs for Frame %d\n", frame);
  {
    int b_row;

    for (b_row = 0; b_row < 4 * rows; ++b_row) {
      int b_col;
      int bindex;

      for (b_col = 0; b_col < 4 * cols; ++b_col) {
        mb_index = (b_row >> 2) * (cols + 1) + (b_col >> 2);
        bindex = (b_row & 3) * 4 + (b_col & 3);

        if (mi[mb_index].mbmi.mode == B_PRED)
          fprintf(mvs, "%2d ", mi[mb_index].bmi[bindex].as_mode);
        else
          fprintf(mvs, "xx ");
      }

      fprintf(mvs, "\n");
    }
  }
  fprintf(mvs, "\n");

  /* print out the macroblock mvs */
  mb_index = 0;
  fprintf(mvs, "MVs for Frame %d\n", frame);

  for (mb_row = 0; mb_row < rows; ++mb_row) {
    for (mb_col = 0; mb_col < cols; ++mb_col) {
      fprintf(mvs, "%5d:%-5d", mi[mb_index].mbmi.mv.as_mv.row / 2,
              mi[mb_index].mbmi.mv.as_mv.col / 2);

      mb_index++;
    }

    mb_index++;
    fprintf(mvs, "\n");
  }

  fprintf(mvs, "\n");

  /* print out the block modes */
  fprintf(mvs, "MVs for Frame %d\n", frame);
  {
    int b_row;

    for (b_row = 0; b_row < 4 * rows; ++b_row) {
      int b_col;
      int bindex;

      for (b_col = 0; b_col < 4 * cols; ++b_col) {
        mb_index = (b_row >> 2) * (cols + 1) + (b_col >> 2);
        bindex = (b_row & 3) * 4 + (b_col & 3);
        fprintf(mvs, "%3d:%-3d ", mi[mb_index].bmi[bindex].mv.as_mv.row,
                mi[mb_index].bmi[bindex].mv.as_mv.col);
      }

      fprintf(mvs, "\n");
    }
  }
  fprintf(mvs, "\n");

  fclose(mvs);
}
