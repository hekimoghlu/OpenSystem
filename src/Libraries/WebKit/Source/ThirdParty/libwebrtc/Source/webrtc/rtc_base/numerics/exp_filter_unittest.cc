/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#include "rtc_base/numerics/exp_filter.h"

#include <cmath>

#include "test/gtest.h"

namespace rtc {

TEST(ExpFilterTest, FirstTimeOutputEqualInput) {
  // No max value defined.
  ExpFilter filter = ExpFilter(0.9f);
  filter.Apply(100.0f, 10.0f);

  // First time, first argument no effect.
  double value = 10.0f;
  EXPECT_FLOAT_EQ(value, filter.filtered());
}

TEST(ExpFilterTest, SecondTime) {
  float value;

  ExpFilter filter = ExpFilter(0.9f);
  filter.Apply(100.0f, 10.0f);

  // First time, first argument no effect.
  value = 10.0f;

  filter.Apply(10.0f, 20.0f);
  float alpha = std::pow(0.9f, 10.0f);
  value = alpha * value + (1.0f - alpha) * 20.0f;
  EXPECT_FLOAT_EQ(value, filter.filtered());
}

TEST(ExpFilterTest, Reset) {
  ExpFilter filter = ExpFilter(0.9f);
  filter.Apply(100.0f, 10.0f);

  filter.Reset(0.8f);
  filter.Apply(100.0f, 1.0f);

  // Become first time after a reset.
  double value = 1.0f;
  EXPECT_FLOAT_EQ(value, filter.filtered());
}

TEST(ExpfilterTest, OutputLimitedByMax) {
  double value;

  // Max value defined.
  ExpFilter filter = ExpFilter(0.9f, 1.0f);
  filter.Apply(100.0f, 10.0f);

  // Limited to max value.
  value = 1.0f;
  EXPECT_EQ(value, filter.filtered());

  filter.Apply(1.0f, 0.0f);
  value = 0.9f * value;
  EXPECT_FLOAT_EQ(value, filter.filtered());
}

}  // namespace rtc
