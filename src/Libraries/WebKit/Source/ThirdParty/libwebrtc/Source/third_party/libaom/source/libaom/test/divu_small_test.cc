/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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
#include <stdlib.h>

#include "gtest/gtest.h"

#include "test/acm_random.h"
#include "aom_dsp/odintrin.h"

using libaom_test::ACMRandom;

TEST(DivuSmallTest, TestDIVUuptoMAX) {
  for (int d = 1; d <= OD_DIVU_DMAX; d++) {
    for (uint32_t x = 1; x <= 1000000; x++) {
      GTEST_ASSERT_EQ(x / d, OD_DIVU_SMALL(x, d))
          << "x=" << x << " d=" << d << " x/d=" << (x / d)
          << " != " << OD_DIVU_SMALL(x, d);
    }
  }
}

TEST(DivuSmallTest, TestDIVUrandI31) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  for (int d = 1; d < OD_DIVU_DMAX; d++) {
    for (int i = 0; i < 1000000; i++) {
      uint32_t x = rnd.Rand31();
      GTEST_ASSERT_EQ(x / d, OD_DIVU_SMALL(x, d))
          << "x=" << x << " d=" << d << " x/d=" << (x / d)
          << " != " << OD_DIVU_SMALL(x, d);
    }
  }
}
