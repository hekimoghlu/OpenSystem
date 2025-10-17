/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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
#include "video/unique_timestamp_counter.h"

#include "test/gtest.h"

namespace webrtc {
namespace {

TEST(UniqueTimestampCounterTest, InitiallyZero) {
  UniqueTimestampCounter counter;
  EXPECT_EQ(counter.GetUniqueSeen(), 0);
}

TEST(UniqueTimestampCounterTest, CountsUniqueValues) {
  UniqueTimestampCounter counter;
  counter.Add(100);
  counter.Add(100);
  counter.Add(200);
  counter.Add(150);
  counter.Add(100);
  EXPECT_EQ(counter.GetUniqueSeen(), 3);
}

TEST(UniqueTimestampCounterTest, ForgetsOldValuesAfter1000NewValues) {
  const int kNumValues = 1500;
  const int kMaxHistory = 1000;
  const uint32_t value = 0xFFFFFFF0;
  UniqueTimestampCounter counter;
  for (int i = 0; i < kNumValues; ++i) {
    counter.Add(value + 10 * i);
  }
  ASSERT_EQ(counter.GetUniqueSeen(), kNumValues);
  // Slightly old values not affect number of seen unique values.
  for (int i = kNumValues - kMaxHistory; i < kNumValues; ++i) {
    counter.Add(value + 10 * i);
  }
  EXPECT_EQ(counter.GetUniqueSeen(), kNumValues);
  // Very old value will be treated as unique.
  counter.Add(value);
  EXPECT_EQ(counter.GetUniqueSeen(), kNumValues + 1);
}

}  // namespace
}  // namespace webrtc
