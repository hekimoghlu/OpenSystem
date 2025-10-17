/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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
#include "rtc_base/data_rate_limiter.h"

#include "test/gtest.h"

namespace rtc {

TEST(RateLimiterTest, TestCanUse) {
  // Diet: Can eat 2,000 calories per day.
  DataRateLimiter limiter = DataRateLimiter(2000, 1.0);

  double monday = 1.0;
  double tuesday = 2.0;
  double thursday = 4.0;

  EXPECT_TRUE(limiter.CanUse(0, monday));
  EXPECT_TRUE(limiter.CanUse(1000, monday));
  EXPECT_TRUE(limiter.CanUse(1999, monday));
  EXPECT_TRUE(limiter.CanUse(2000, monday));
  EXPECT_FALSE(limiter.CanUse(2001, monday));

  limiter.Use(1000, monday);

  EXPECT_TRUE(limiter.CanUse(0, monday));
  EXPECT_TRUE(limiter.CanUse(999, monday));
  EXPECT_TRUE(limiter.CanUse(1000, monday));
  EXPECT_FALSE(limiter.CanUse(1001, monday));

  limiter.Use(1000, monday);

  EXPECT_TRUE(limiter.CanUse(0, monday));
  EXPECT_FALSE(limiter.CanUse(1, monday));

  EXPECT_TRUE(limiter.CanUse(0, tuesday));
  EXPECT_TRUE(limiter.CanUse(1, tuesday));
  EXPECT_TRUE(limiter.CanUse(1999, tuesday));
  EXPECT_TRUE(limiter.CanUse(2000, tuesday));
  EXPECT_FALSE(limiter.CanUse(2001, tuesday));

  limiter.Use(1000, tuesday);

  EXPECT_TRUE(limiter.CanUse(1000, tuesday));
  EXPECT_FALSE(limiter.CanUse(1001, tuesday));

  limiter.Use(1000, thursday);

  EXPECT_TRUE(limiter.CanUse(1000, tuesday));
  EXPECT_FALSE(limiter.CanUse(1001, tuesday));
}

}  // namespace rtc
