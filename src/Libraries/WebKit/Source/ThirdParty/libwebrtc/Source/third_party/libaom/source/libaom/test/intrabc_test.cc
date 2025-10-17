/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#include "gtest/gtest.h"

#include "config/aom_config.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/enums.h"
#include "av1/common/mv.h"
#include "av1/common/mvref_common.h"
#include "av1/common/tile_common.h"

namespace {
TEST(IntrabcTest, DvValidation) {
  struct DvTestCase {
    MV dv;
    int mi_row_offset;
    int mi_col_offset;
    BLOCK_SIZE bsize;
    bool valid;
  };
  const int kSubPelScale = 8;
  const int kTileMaxMibWidth = 8;
  const DvTestCase kDvCases[] = {
    { { 0, 0 }, 0, 0, BLOCK_128X128, false },
    { { 0, 0 }, 0, 0, BLOCK_64X64, false },
    { { 0, 0 }, 0, 0, BLOCK_32X32, false },
    { { 0, 0 }, 0, 0, BLOCK_16X16, false },
    { { 0, 0 }, 0, 0, BLOCK_8X8, false },
    { { 0, 0 }, 0, 0, BLOCK_4X4, false },
    { { -MAX_SB_SIZE * kSubPelScale, -MAX_SB_SIZE * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_16X16,
      true },
    { { 0, -MAX_SB_SIZE * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_16X16,
      false },
    { { -MAX_SB_SIZE * kSubPelScale, 0 },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_16X16,
      true },
    { { MAX_SB_SIZE * kSubPelScale, 0 },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_16X16,
      false },
    { { 0, MAX_SB_SIZE * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_16X16,
      false },
    { { -32 * kSubPelScale, -32 * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_32X32,
      true },
    { { -32 * kSubPelScale, -32 * kSubPelScale },
      32 / MI_SIZE,
      32 / MI_SIZE,
      BLOCK_32X32,
      false },
    { { -32 * kSubPelScale - kSubPelScale / 2, -32 * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_32X32,
      false },
    { { -33 * kSubPelScale, -32 * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_32X32,
      true },
    { { -32 * kSubPelScale, -32 * kSubPelScale - kSubPelScale / 2 },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_32X32,
      false },
    { { -32 * kSubPelScale, -33 * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_32X32,
      true },
    { { -MAX_SB_SIZE * kSubPelScale, -MAX_SB_SIZE * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_LARGEST,
      true },
    { { -(MAX_SB_SIZE + 1) * kSubPelScale, -MAX_SB_SIZE * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_LARGEST,
      false },
    { { -MAX_SB_SIZE * kSubPelScale, -(MAX_SB_SIZE + 1) * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_LARGEST,
      false },
    { { -(MAX_SB_SIZE - 1) * kSubPelScale, -MAX_SB_SIZE * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_LARGEST,
      false },
    { { -MAX_SB_SIZE * kSubPelScale, -(MAX_SB_SIZE - 1) * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_LARGEST,
      true },
    { { -(MAX_SB_SIZE - 1) * kSubPelScale, -(MAX_SB_SIZE - 1) * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_LARGEST,
      false },
    { { -MAX_SB_SIZE * kSubPelScale, MAX_SB_SIZE * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_LARGEST,
      false },
    { { -MAX_SB_SIZE * kSubPelScale,
        (kTileMaxMibWidth - 2) * MAX_SB_SIZE * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_LARGEST,
      false },
    { { -MAX_SB_SIZE * kSubPelScale,
        ((kTileMaxMibWidth - 2) * MAX_SB_SIZE + 1) * kSubPelScale },
      MAX_SB_SIZE / MI_SIZE,
      MAX_SB_SIZE / MI_SIZE,
      BLOCK_LARGEST,
      false },
  };

  MACROBLOCKD xd;
  memset(&xd, 0, sizeof(xd));
  xd.tile.mi_row_start = 8 * MAX_MIB_SIZE;
  xd.tile.mi_row_end = 16 * MAX_MIB_SIZE;
  xd.tile.mi_col_start = 24 * MAX_MIB_SIZE;
  xd.tile.mi_col_end = xd.tile.mi_col_start + kTileMaxMibWidth * MAX_MIB_SIZE;
  xd.plane[1].subsampling_x = 1;
  xd.plane[1].subsampling_y = 1;
  xd.plane[2].subsampling_x = 1;
  xd.plane[2].subsampling_y = 1;

  SequenceHeader seq_params = {};
  AV1_COMMON cm;
  memset(&cm, 0, sizeof(cm));
  cm.seq_params = &seq_params;

  for (const DvTestCase &dv_case : kDvCases) {
    const int mi_row = xd.tile.mi_row_start + dv_case.mi_row_offset;
    const int mi_col = xd.tile.mi_col_start + dv_case.mi_col_offset;
    xd.is_chroma_ref = is_chroma_reference(mi_row, mi_col, dv_case.bsize,
                                           xd.plane[1].subsampling_x,
                                           xd.plane[1].subsampling_y);
    EXPECT_EQ(static_cast<int>(dv_case.valid),
              av1_is_dv_valid(dv_case.dv, &cm, &xd, mi_row, mi_col,
                              dv_case.bsize, MAX_MIB_SIZE_LOG2));
  }
}
}  // namespace
