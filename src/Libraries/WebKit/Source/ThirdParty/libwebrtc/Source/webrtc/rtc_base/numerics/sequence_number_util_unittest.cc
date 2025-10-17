/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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
#include "rtc_base/numerics/sequence_number_util.h"

#include <cstdint>
#include <iterator>
#include <set>

#include "rtc_base/numerics/mod_ops.h"
#include "test/gtest.h"

namespace webrtc {
class TestSeqNumUtil : public ::testing::Test {
 protected:
  // Can't use std::numeric_limits<unsigned long>::max() since
  // MSVC doesn't support constexpr.
  static const unsigned long ulmax = ~0ul;  // NOLINT
};

TEST_F(TestSeqNumUtil, AheadOrAt) {
  uint8_t x = 0;
  uint8_t y = 0;
  ASSERT_TRUE(AheadOrAt(x, y));
  ++x;
  ASSERT_TRUE(AheadOrAt(x, y));
  ASSERT_FALSE(AheadOrAt(y, x));
  for (int i = 0; i < 256; ++i) {
    ASSERT_TRUE(AheadOrAt(x, y));
    ++x;
    ++y;
  }

  x = 128;
  y = 0;
  ASSERT_TRUE(AheadOrAt(x, y));
  ASSERT_FALSE(AheadOrAt(y, x));

  x = 129;
  ASSERT_FALSE(AheadOrAt(x, y));
  ASSERT_TRUE(AheadOrAt(y, x));
  ASSERT_TRUE(AheadOrAt<uint16_t>(x, y));
  ASSERT_FALSE(AheadOrAt<uint16_t>(y, x));
}

TEST_F(TestSeqNumUtil, AheadOrAtWithDivisor) {
  ASSERT_TRUE((AheadOrAt<uint8_t, 11>(5, 0)));
  ASSERT_FALSE((AheadOrAt<uint8_t, 11>(6, 0)));
  ASSERT_FALSE((AheadOrAt<uint8_t, 11>(0, 5)));
  ASSERT_TRUE((AheadOrAt<uint8_t, 11>(0, 6)));

  ASSERT_TRUE((AheadOrAt<uint8_t, 10>(5, 0)));
  ASSERT_FALSE((AheadOrAt<uint8_t, 10>(6, 0)));
  ASSERT_FALSE((AheadOrAt<uint8_t, 10>(0, 5)));
  ASSERT_TRUE((AheadOrAt<uint8_t, 10>(0, 6)));

  const uint8_t D = 211;
  uint8_t x = 0;
  for (int i = 0; i < D; ++i) {
    uint8_t next_x = Add<D>(x, 1);
    ASSERT_TRUE((AheadOrAt<uint8_t, D>(i, i)));
    ASSERT_TRUE((AheadOrAt<uint8_t, D>(next_x, i)));
    ASSERT_FALSE((AheadOrAt<uint8_t, D>(i, next_x)));
    x = next_x;
  }
}

TEST_F(TestSeqNumUtil, AheadOf) {
  uint8_t x = 0;
  uint8_t y = 0;
  ASSERT_FALSE(AheadOf(x, y));
  ++x;
  ASSERT_TRUE(AheadOf(x, y));
  ASSERT_FALSE(AheadOf(y, x));
  for (int i = 0; i < 256; ++i) {
    ASSERT_TRUE(AheadOf(x, y));
    ++x;
    ++y;
  }

  x = 128;
  y = 0;
  for (int i = 0; i < 128; ++i) {
    ASSERT_TRUE(AheadOf(x, y));
    ASSERT_FALSE(AheadOf(y, x));
    x++;
    y++;
  }

  for (int i = 0; i < 128; ++i) {
    ASSERT_FALSE(AheadOf(x, y));
    ASSERT_TRUE(AheadOf(y, x));
    x++;
    y++;
  }

  x = 129;
  y = 0;
  ASSERT_FALSE(AheadOf(x, y));
  ASSERT_TRUE(AheadOf(y, x));
  ASSERT_TRUE(AheadOf<uint16_t>(x, y));
  ASSERT_FALSE(AheadOf<uint16_t>(y, x));
}

TEST_F(TestSeqNumUtil, AheadOfWithDivisor) {
  ASSERT_TRUE((AheadOf<uint8_t, 11>(5, 0)));
  ASSERT_FALSE((AheadOf<uint8_t, 11>(6, 0)));
  ASSERT_FALSE((AheadOf<uint8_t, 11>(0, 5)));
  ASSERT_TRUE((AheadOf<uint8_t, 11>(0, 6)));

  ASSERT_TRUE((AheadOf<uint8_t, 10>(5, 0)));
  ASSERT_FALSE((AheadOf<uint8_t, 10>(6, 0)));
  ASSERT_FALSE((AheadOf<uint8_t, 10>(0, 5)));
  ASSERT_TRUE((AheadOf<uint8_t, 10>(0, 6)));

  const uint8_t D = 211;
  uint8_t x = 0;
  for (int i = 0; i < D; ++i) {
    uint8_t next_x = Add<D>(x, 1);
    ASSERT_FALSE((AheadOf<uint8_t, D>(i, i)));
    ASSERT_TRUE((AheadOf<uint8_t, D>(next_x, i)));
    ASSERT_FALSE((AheadOf<uint8_t, D>(i, next_x)));
    x = next_x;
  }
}

TEST_F(TestSeqNumUtil, ForwardDiffWithDivisor) {
  const uint8_t kDivisor = 211;

  for (uint8_t i = 0; i < kDivisor - 1; ++i) {
    ASSERT_EQ(0, (ForwardDiff<uint8_t, kDivisor>(i, i)));
    ASSERT_EQ(1, (ForwardDiff<uint8_t, kDivisor>(i, i + 1)));
    ASSERT_EQ(kDivisor - 1, (ForwardDiff<uint8_t, kDivisor>(i + 1, i)));
  }

  for (uint8_t i = 1; i < kDivisor; ++i) {
    ASSERT_EQ(i, (ForwardDiff<uint8_t, kDivisor>(0, i)));
    ASSERT_EQ(kDivisor - i, (ForwardDiff<uint8_t, kDivisor>(i, 0)));
  }
}

TEST_F(TestSeqNumUtil, ReverseDiffWithDivisor) {
  const uint8_t kDivisor = 241;

  for (uint8_t i = 0; i < kDivisor - 1; ++i) {
    ASSERT_EQ(0, (ReverseDiff<uint8_t, kDivisor>(i, i)));
    ASSERT_EQ(kDivisor - 1, (ReverseDiff<uint8_t, kDivisor>(i, i + 1)));
    ASSERT_EQ(1, (ReverseDiff<uint8_t, kDivisor>(i + 1, i)));
  }

  for (uint8_t i = 1; i < kDivisor; ++i) {
    ASSERT_EQ(kDivisor - i, (ReverseDiff<uint8_t, kDivisor>(0, i)));
    ASSERT_EQ(i, (ReverseDiff<uint8_t, kDivisor>(i, 0)));
  }
}

TEST_F(TestSeqNumUtil, SeqNumComparator) {
  std::set<uint8_t, AscendingSeqNumComp<uint8_t>> seq_nums_asc;
  std::set<uint8_t, DescendingSeqNumComp<uint8_t>> seq_nums_desc;

  uint8_t x = 0;
  for (int i = 0; i < 128; ++i) {
    seq_nums_asc.insert(x);
    seq_nums_desc.insert(x);
    ASSERT_EQ(x, *seq_nums_asc.begin());
    ASSERT_EQ(x, *seq_nums_desc.rbegin());
    ++x;
  }

  seq_nums_asc.clear();
  seq_nums_desc.clear();
  x = 199;
  for (int i = 0; i < 128; ++i) {
    seq_nums_asc.insert(x);
    seq_nums_desc.insert(x);
    ASSERT_EQ(x, *seq_nums_asc.begin());
    ASSERT_EQ(x, *seq_nums_desc.rbegin());
    ++x;
  }
}

TEST_F(TestSeqNumUtil, SeqNumComparatorWithDivisor) {
  const uint8_t D = 223;

  std::set<uint8_t, AscendingSeqNumComp<uint8_t, D>> seq_nums_asc;
  std::set<uint8_t, DescendingSeqNumComp<uint8_t, D>> seq_nums_desc;

  uint8_t x = 0;
  for (int i = 0; i < D / 2; ++i) {
    seq_nums_asc.insert(x);
    seq_nums_desc.insert(x);
    ASSERT_EQ(x, *seq_nums_asc.begin());
    ASSERT_EQ(x, *seq_nums_desc.rbegin());
    x = Add<D>(x, 1);
  }

  seq_nums_asc.clear();
  seq_nums_desc.clear();
  x = 200;
  for (int i = 0; i < D / 2; ++i) {
    seq_nums_asc.insert(x);
    seq_nums_desc.insert(x);
    ASSERT_EQ(x, *seq_nums_asc.begin());
    ASSERT_EQ(x, *seq_nums_desc.rbegin());
    x = Add<D>(x, 1);
  }
}

}  // namespace webrtc
