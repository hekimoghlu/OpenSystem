/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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
#include "net/dcsctp/public/types.h"

#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {

TEST(TypesTest, DurationOperators) {
  DurationMs d1(10);
  DurationMs d2(25);
  EXPECT_EQ(d1 + d2, DurationMs(35));
  EXPECT_EQ(d2 - d1, DurationMs(15));

  d1 += d2;
  EXPECT_EQ(d1, DurationMs(35));

  d1 -= DurationMs(5);
  EXPECT_EQ(d1, DurationMs(30));

  d1 *= 1.5;
  EXPECT_EQ(d1, DurationMs(45));

  EXPECT_EQ(DurationMs(10) * 2, DurationMs(20));
}

TEST(TypesTest, TimeOperators) {
  EXPECT_EQ(TimeMs(250) + DurationMs(100), TimeMs(350));
  EXPECT_EQ(DurationMs(250) + TimeMs(100), TimeMs(350));
  EXPECT_EQ(TimeMs(250) - DurationMs(100), TimeMs(150));
  EXPECT_EQ(TimeMs(250) - TimeMs(100), DurationMs(150));

  TimeMs t1(150);
  t1 -= DurationMs(50);
  EXPECT_EQ(t1, TimeMs(100));
  t1 += DurationMs(200);
  EXPECT_EQ(t1, TimeMs(300));
}

}  // namespace
}  // namespace dcsctp
