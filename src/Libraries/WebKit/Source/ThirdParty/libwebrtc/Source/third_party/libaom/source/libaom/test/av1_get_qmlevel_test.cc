/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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

#include "av1/common/quant_common.h"

namespace {

TEST(GetQmLevelTest, Regular) {
  // Get some QM levels from representative qindex values.
  int qindex_1_qmlevel = aom_get_qmlevel(1, DEFAULT_QM_FIRST, DEFAULT_QM_LAST);
  int qindex_60_qmlevel =
      aom_get_qmlevel(60, DEFAULT_QM_FIRST, DEFAULT_QM_LAST);
  int qindex_120_qmlevel =
      aom_get_qmlevel(120, DEFAULT_QM_FIRST, DEFAULT_QM_LAST);
  int qindex_180_qmlevel =
      aom_get_qmlevel(180, DEFAULT_QM_FIRST, DEFAULT_QM_LAST);
  int qindex_255_qmlevel =
      aom_get_qmlevel(255, DEFAULT_QM_FIRST, DEFAULT_QM_LAST);

  // Extreme qindex values should also result in extreme QM levels.
  EXPECT_EQ(qindex_1_qmlevel, DEFAULT_QM_FIRST);
  EXPECT_EQ(qindex_255_qmlevel, DEFAULT_QM_LAST);

  // aom_get_qmlevel() QMs become steeper (i.e. QM levels become lower) the
  // lower the qindex.
  EXPECT_GE(qindex_255_qmlevel, qindex_180_qmlevel);
  EXPECT_GE(qindex_180_qmlevel, qindex_120_qmlevel);
  EXPECT_GE(qindex_120_qmlevel, qindex_60_qmlevel);
  EXPECT_GE(qindex_60_qmlevel, qindex_1_qmlevel);
  EXPECT_GT(qindex_255_qmlevel, qindex_1_qmlevel);

  // Set min and max QM levels to be lower than DEFAULT_QM_FIRST.
  int qindex_1_qmlevel_belowfirst = aom_get_qmlevel(1, 1, DEFAULT_QM_FIRST - 1);
  int qindex_255_qmlevel_belowfirst =
      aom_get_qmlevel(255, 1, DEFAULT_QM_FIRST - 1);

  // Formula should always respect QM level boundaries, even when they're below
  // DEFAULT_QM_FIRST.
  EXPECT_LT(qindex_1_qmlevel_belowfirst, DEFAULT_QM_FIRST);
  EXPECT_LT(qindex_255_qmlevel_belowfirst, DEFAULT_QM_FIRST);

  // Set min and max QM levels to be higher than DEFAULT_QM_LAST.
  int qindex_1_qmlevel_abovelast = aom_get_qmlevel(1, DEFAULT_QM_LAST + 1, 15);
  int qindex_255_qmlevel_abovelast =
      aom_get_qmlevel(255, DEFAULT_QM_LAST + 1, 15);

  // Formula should always respect QM level boundaries, even when they're above
  // DEFAULT_QM_LAST.
  EXPECT_GT(qindex_1_qmlevel_abovelast, DEFAULT_QM_LAST);
  EXPECT_GT(qindex_255_qmlevel_abovelast, DEFAULT_QM_LAST);
}

TEST(GetQmLevelTest, AllIntra) {
  // Get some QM levels from representative qindex values.
  int qindex_1_qmlevel = aom_get_qmlevel_allintra(1, DEFAULT_QM_FIRST_ALLINTRA,
                                                  DEFAULT_QM_LAST_ALLINTRA);
  int qindex_60_qmlevel = aom_get_qmlevel_allintra(
      60, DEFAULT_QM_FIRST_ALLINTRA, DEFAULT_QM_LAST_ALLINTRA);
  int qindex_120_qmlevel = aom_get_qmlevel_allintra(
      120, DEFAULT_QM_FIRST_ALLINTRA, DEFAULT_QM_LAST_ALLINTRA);
  int qindex_180_qmlevel = aom_get_qmlevel_allintra(
      180, DEFAULT_QM_FIRST_ALLINTRA, DEFAULT_QM_LAST_ALLINTRA);
  int qindex_255_qmlevel = aom_get_qmlevel_allintra(
      255, DEFAULT_QM_FIRST_ALLINTRA, DEFAULT_QM_LAST_ALLINTRA);

  // Extreme qindex values should also result in extreme QM levels.
  EXPECT_EQ(qindex_1_qmlevel, DEFAULT_QM_LAST_ALLINTRA);
  EXPECT_EQ(qindex_255_qmlevel, DEFAULT_QM_FIRST_ALLINTRA);

  // Unlike with aom_get_qmlevel(), aom_get_qmlevel_allintra() QMs become
  // flatter (i.e. QM levels become higher) the lower the qindex.
  EXPECT_LE(qindex_255_qmlevel, qindex_180_qmlevel);
  EXPECT_LE(qindex_180_qmlevel, qindex_120_qmlevel);
  EXPECT_LE(qindex_120_qmlevel, qindex_60_qmlevel);
  EXPECT_LE(qindex_60_qmlevel, qindex_1_qmlevel);
  EXPECT_LT(qindex_255_qmlevel, qindex_1_qmlevel);

  // Set min and max QM levels to be lower than DEFAULT_QM_FIRST_ALLINTRA.
  int qindex_1_qmlevel_belowfirst =
      aom_get_qmlevel_allintra(1, 1, DEFAULT_QM_FIRST_ALLINTRA - 1);
  int qindex_255_qmlevel_belowfirst =
      aom_get_qmlevel_allintra(255, 1, DEFAULT_QM_FIRST_ALLINTRA - 1);

  // Formula should always respect QM level boundaries, even when they're below
  // DEFAULT_QM_FIRST_ALLINTRA.
  EXPECT_EQ(qindex_1_qmlevel_belowfirst, DEFAULT_QM_FIRST_ALLINTRA - 1);
  EXPECT_EQ(qindex_255_qmlevel_belowfirst, DEFAULT_QM_FIRST_ALLINTRA - 1);

  // Set min and max QM levels to be higher than DEFAULT_QM_LAST_ALLINTRA.
  int qindex_1_qmlevel_abovelast =
      aom_get_qmlevel_allintra(1, DEFAULT_QM_LAST_ALLINTRA + 1, 15);
  int qindex_255_qmlevel_abovelast =
      aom_get_qmlevel_allintra(255, DEFAULT_QM_LAST_ALLINTRA + 1, 15);

  // Formula should always respect QM level boundaries, even when they're above
  // DEFAULT_QM_LAST_ALLINTRA.
  EXPECT_EQ(qindex_1_qmlevel_abovelast, DEFAULT_QM_LAST_ALLINTRA + 1);
  EXPECT_EQ(qindex_255_qmlevel_abovelast, DEFAULT_QM_LAST_ALLINTRA + 1);
}

}  // namespace
