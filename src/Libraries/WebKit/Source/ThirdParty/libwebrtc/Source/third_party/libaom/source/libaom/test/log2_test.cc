/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#include <limits.h>
#include <math.h>

#include "aom_ports/bitops.h"
#include "av1/common/entropymode.h"
#include "gtest/gtest.h"

TEST(Log2Test, GetMsb) {
  // Test small numbers exhaustively.
  for (unsigned int n = 1; n < 10000; n++) {
    EXPECT_EQ(get_msb(n), static_cast<int>(floor(log2(n))));
  }

  // Test every power of 2 and the two adjacent numbers.
  for (int exponent = 2; exponent < 32; exponent++) {
    const unsigned int power_of_2 = 1U << exponent;
    EXPECT_EQ(get_msb(power_of_2 - 1), exponent - 1);
    EXPECT_EQ(get_msb(power_of_2), exponent);
    EXPECT_EQ(get_msb(power_of_2 + 1), exponent);
  }
}

TEST(Log2Test, Av1CeilLog2) {
  // Test small numbers exhaustively.
  EXPECT_EQ(av1_ceil_log2(0), 0);
  for (int n = 1; n < 10000; n++) {
    EXPECT_EQ(av1_ceil_log2(n), static_cast<int>(ceil(log2(n))));
  }

  // Test every power of 2 and the two adjacent numbers.
  for (int exponent = 2; exponent < 31; exponent++) {
    const int power_of_2 = 1 << exponent;
    EXPECT_EQ(av1_ceil_log2(power_of_2 - 1), exponent);
    EXPECT_EQ(av1_ceil_log2(power_of_2), exponent);
    EXPECT_EQ(av1_ceil_log2(power_of_2 + 1), exponent + 1);
  }

  // INT_MAX = 2^31 - 1
  EXPECT_EQ(av1_ceil_log2(INT_MAX), 31);
}
