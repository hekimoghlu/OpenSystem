/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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
#include <math.h>
#include <vector>

#include "av1/common/quant_common.h"
#include "av1/encoder/rd.h"
#include "aom/aom_codec.h"
#include "gtest/gtest.h"

namespace {

TEST(RdTest, GetDeltaqOffsetValueTest1) {
  aom_bit_depth_t bit_depth = AOM_BITS_8;
  double beta = 4;
  int q_index = 29;
  int dc_q_step =
      av1_dc_quant_QTX(q_index, 0, static_cast<aom_bit_depth_t>(bit_depth));
  EXPECT_EQ(dc_q_step, 32);

  int ref_new_dc_q_step = static_cast<int>(round(dc_q_step / sqrt(beta)));
  EXPECT_EQ(ref_new_dc_q_step, 16);

  int delta_q = av1_get_deltaq_offset(bit_depth, q_index, beta);
  int new_dc_q_step = av1_dc_quant_QTX(q_index, delta_q,
                                       static_cast<aom_bit_depth_t>(bit_depth));

  EXPECT_EQ(new_dc_q_step, ref_new_dc_q_step);
}

TEST(RdTest, GetDeltaqOffsetValueTest2) {
  aom_bit_depth_t bit_depth = AOM_BITS_8;
  double beta = 1.0 / 4.0;
  int q_index = 29;
  int dc_q_step =
      av1_dc_quant_QTX(q_index, 0, static_cast<aom_bit_depth_t>(bit_depth));
  EXPECT_EQ(dc_q_step, 32);

  int ref_new_dc_q_step = static_cast<int>(round(dc_q_step / sqrt(beta)));
  EXPECT_EQ(ref_new_dc_q_step, 64);

  int delta_q = av1_get_deltaq_offset(bit_depth, q_index, beta);
  int new_dc_q_step = av1_dc_quant_QTX(q_index, delta_q,
                                       static_cast<aom_bit_depth_t>(bit_depth));

  EXPECT_EQ(new_dc_q_step, ref_new_dc_q_step);
}

TEST(RdTest, GetDeltaqOffsetBoundaryTest1) {
  aom_bit_depth_t bit_depth = AOM_BITS_8;
  double beta = 0.000000001;
  std::vector<int> q_index_ls = { 254, 255 };
  for (auto q_index : q_index_ls) {
    int delta_q = av1_get_deltaq_offset(bit_depth, q_index, beta);
    EXPECT_EQ(q_index + delta_q, 255);
  }
}

TEST(RdTest, GetDeltaqOffsetBoundaryTest2) {
  aom_bit_depth_t bit_depth = AOM_BITS_8;
  double beta = 100;
  std::vector<int> q_index_ls = { 1, 0 };
  for (auto q_index : q_index_ls) {
    int delta_q = av1_get_deltaq_offset(bit_depth, q_index, beta);
    EXPECT_EQ(q_index + delta_q, 0);
  }
}

TEST(RdTest, GetDeltaqOffsetUnitaryTest1) {
  aom_bit_depth_t bit_depth = AOM_BITS_8;
  double beta = 1;
  for (int q_index = 0; q_index < 255; ++q_index) {
    int delta_q = av1_get_deltaq_offset(bit_depth, q_index, beta);
    EXPECT_EQ(delta_q, 0);
  }
}

}  // namespace
