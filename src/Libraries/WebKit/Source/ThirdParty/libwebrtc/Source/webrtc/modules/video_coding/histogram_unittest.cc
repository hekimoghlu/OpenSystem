/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
#include "modules/video_coding/histogram.h"

#include "test/gtest.h"

namespace webrtc {
namespace video_coding {

class TestHistogram : public ::testing::Test {
 protected:
  TestHistogram() : histogram_(5, 10) {}
  Histogram histogram_;
};

TEST_F(TestHistogram, NumValues) {
  EXPECT_EQ(0ul, histogram_.NumValues());
  histogram_.Add(0);
  EXPECT_EQ(1ul, histogram_.NumValues());
}

TEST_F(TestHistogram, InverseCdf) {
  histogram_.Add(0);
  histogram_.Add(1);
  histogram_.Add(2);
  histogram_.Add(3);
  histogram_.Add(4);
  EXPECT_EQ(5ul, histogram_.NumValues());
  EXPECT_EQ(1ul, histogram_.InverseCdf(0.2f));
  EXPECT_EQ(2ul, histogram_.InverseCdf(0.2000001f));
  EXPECT_EQ(4ul, histogram_.InverseCdf(0.8f));

  histogram_.Add(0);
  EXPECT_EQ(6ul, histogram_.NumValues());
  EXPECT_EQ(1ul, histogram_.InverseCdf(0.2f));
  EXPECT_EQ(1ul, histogram_.InverseCdf(0.2000001f));
}

TEST_F(TestHistogram, ReplaceOldValues) {
  histogram_.Add(0);
  histogram_.Add(0);
  histogram_.Add(0);
  histogram_.Add(0);
  histogram_.Add(0);
  histogram_.Add(1);
  histogram_.Add(1);
  histogram_.Add(1);
  histogram_.Add(1);
  histogram_.Add(1);
  EXPECT_EQ(10ul, histogram_.NumValues());
  EXPECT_EQ(1ul, histogram_.InverseCdf(0.5f));
  EXPECT_EQ(2ul, histogram_.InverseCdf(0.5000001f));

  histogram_.Add(4);
  histogram_.Add(4);
  histogram_.Add(4);
  histogram_.Add(4);
  EXPECT_EQ(10ul, histogram_.NumValues());
  EXPECT_EQ(1ul, histogram_.InverseCdf(0.1f));
  EXPECT_EQ(2ul, histogram_.InverseCdf(0.5f));

  histogram_.Add(20);
  EXPECT_EQ(10ul, histogram_.NumValues());
  EXPECT_EQ(2ul, histogram_.InverseCdf(0.5f));
  EXPECT_EQ(5ul, histogram_.InverseCdf(0.5000001f));
}

}  // namespace video_coding
}  // namespace webrtc
