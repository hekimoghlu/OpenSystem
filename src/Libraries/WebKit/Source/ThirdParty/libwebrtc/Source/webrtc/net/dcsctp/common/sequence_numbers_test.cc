/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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
#include "net/dcsctp/common/sequence_numbers.h"

#include "test/gmock.h"

namespace dcsctp {
namespace {

using Wrapped = webrtc::StrongAlias<class WrappedTag, uint16_t>;
using TestSequence = UnwrappedSequenceNumber<Wrapped>;

TEST(SequenceNumbersTest, SimpleUnwrapping) {
  TestSequence::Unwrapper unwrapper;

  TestSequence s0 = unwrapper.Unwrap(Wrapped(0));
  TestSequence s1 = unwrapper.Unwrap(Wrapped(1));
  TestSequence s2 = unwrapper.Unwrap(Wrapped(2));
  TestSequence s3 = unwrapper.Unwrap(Wrapped(3));

  EXPECT_LT(s0, s1);
  EXPECT_LT(s0, s2);
  EXPECT_LT(s0, s3);
  EXPECT_LT(s1, s2);
  EXPECT_LT(s1, s3);
  EXPECT_LT(s2, s3);

  EXPECT_EQ(TestSequence::Difference(s1, s0), 1);
  EXPECT_EQ(TestSequence::Difference(s2, s0), 2);
  EXPECT_EQ(TestSequence::Difference(s3, s0), 3);

  EXPECT_GT(s1, s0);
  EXPECT_GT(s2, s0);
  EXPECT_GT(s3, s0);
  EXPECT_GT(s2, s1);
  EXPECT_GT(s3, s1);
  EXPECT_GT(s3, s2);

  s0.Increment();
  EXPECT_EQ(s0, s1);
  s1.Increment();
  EXPECT_EQ(s1, s2);
  s2.Increment();
  EXPECT_EQ(s2, s3);

  EXPECT_EQ(TestSequence::AddTo(s0, 2), s3);
}

TEST(SequenceNumbersTest, MidValueUnwrapping) {
  TestSequence::Unwrapper unwrapper;

  TestSequence s0 = unwrapper.Unwrap(Wrapped(0x7FFE));
  TestSequence s1 = unwrapper.Unwrap(Wrapped(0x7FFF));
  TestSequence s2 = unwrapper.Unwrap(Wrapped(0x8000));
  TestSequence s3 = unwrapper.Unwrap(Wrapped(0x8001));

  EXPECT_LT(s0, s1);
  EXPECT_LT(s0, s2);
  EXPECT_LT(s0, s3);
  EXPECT_LT(s1, s2);
  EXPECT_LT(s1, s3);
  EXPECT_LT(s2, s3);

  EXPECT_EQ(TestSequence::Difference(s1, s0), 1);
  EXPECT_EQ(TestSequence::Difference(s2, s0), 2);
  EXPECT_EQ(TestSequence::Difference(s3, s0), 3);

  EXPECT_GT(s1, s0);
  EXPECT_GT(s2, s0);
  EXPECT_GT(s3, s0);
  EXPECT_GT(s2, s1);
  EXPECT_GT(s3, s1);
  EXPECT_GT(s3, s2);

  s0.Increment();
  EXPECT_EQ(s0, s1);
  s1.Increment();
  EXPECT_EQ(s1, s2);
  s2.Increment();
  EXPECT_EQ(s2, s3);

  EXPECT_EQ(TestSequence::AddTo(s0, 2), s3);
}

TEST(SequenceNumbersTest, WrappedUnwrapping) {
  TestSequence::Unwrapper unwrapper;

  TestSequence s0 = unwrapper.Unwrap(Wrapped(0xFFFE));
  TestSequence s1 = unwrapper.Unwrap(Wrapped(0xFFFF));
  TestSequence s2 = unwrapper.Unwrap(Wrapped(0x0000));
  TestSequence s3 = unwrapper.Unwrap(Wrapped(0x0001));

  EXPECT_LT(s0, s1);
  EXPECT_LT(s0, s2);
  EXPECT_LT(s0, s3);
  EXPECT_LT(s1, s2);
  EXPECT_LT(s1, s3);
  EXPECT_LT(s2, s3);

  EXPECT_EQ(TestSequence::Difference(s1, s0), 1);
  EXPECT_EQ(TestSequence::Difference(s2, s0), 2);
  EXPECT_EQ(TestSequence::Difference(s3, s0), 3);

  EXPECT_GT(s1, s0);
  EXPECT_GT(s2, s0);
  EXPECT_GT(s3, s0);
  EXPECT_GT(s2, s1);
  EXPECT_GT(s3, s1);
  EXPECT_GT(s3, s2);

  s0.Increment();
  EXPECT_EQ(s0, s1);
  s1.Increment();
  EXPECT_EQ(s1, s2);
  s2.Increment();
  EXPECT_EQ(s2, s3);

  EXPECT_EQ(TestSequence::AddTo(s0, 2), s3);
}

TEST(SequenceNumbersTest, WrapAroundAFewTimes) {
  TestSequence::Unwrapper unwrapper;

  TestSequence s0 = unwrapper.Unwrap(Wrapped(0));
  TestSequence prev = s0;

  for (uint32_t i = 1; i < 65536 * 3; i++) {
    uint16_t wrapped = static_cast<uint16_t>(i);
    TestSequence si = unwrapper.Unwrap(Wrapped(wrapped));

    EXPECT_LT(s0, si);
    EXPECT_LT(prev, si);
    prev = si;
  }
}

TEST(SequenceNumbersTest, IncrementIsSameAsWrapped) {
  TestSequence::Unwrapper unwrapper;

  TestSequence s0 = unwrapper.Unwrap(Wrapped(0));

  for (uint32_t i = 1; i < 65536 * 2; i++) {
    uint16_t wrapped = static_cast<uint16_t>(i);
    TestSequence si = unwrapper.Unwrap(Wrapped(wrapped));

    s0.Increment();
    EXPECT_EQ(s0, si);
  }
}

TEST(SequenceNumbersTest, UnwrappingLargerNumberIsAlwaysLarger) {
  TestSequence::Unwrapper unwrapper;

  for (uint32_t i = 1; i < 65536 * 2; i++) {
    uint16_t wrapped = static_cast<uint16_t>(i);
    TestSequence si = unwrapper.Unwrap(Wrapped(wrapped));

    EXPECT_GT(unwrapper.Unwrap(Wrapped(wrapped + 1)), si);
    EXPECT_GT(unwrapper.Unwrap(Wrapped(wrapped + 5)), si);
    EXPECT_GT(unwrapper.Unwrap(Wrapped(wrapped + 10)), si);
    EXPECT_GT(unwrapper.Unwrap(Wrapped(wrapped + 100)), si);
  }
}

TEST(SequenceNumbersTest, UnwrappingSmallerNumberIsAlwaysSmaller) {
  TestSequence::Unwrapper unwrapper;

  for (uint32_t i = 1; i < 65536 * 2; i++) {
    uint16_t wrapped = static_cast<uint16_t>(i);
    TestSequence si = unwrapper.Unwrap(Wrapped(wrapped));

    EXPECT_LT(unwrapper.Unwrap(Wrapped(wrapped - 1)), si);
    EXPECT_LT(unwrapper.Unwrap(Wrapped(wrapped - 5)), si);
    EXPECT_LT(unwrapper.Unwrap(Wrapped(wrapped - 10)), si);
    EXPECT_LT(unwrapper.Unwrap(Wrapped(wrapped - 100)), si);
  }
}

TEST(SequenceNumbersTest, DifferenceIsAbsolute) {
  TestSequence::Unwrapper unwrapper;

  TestSequence this_value = unwrapper.Unwrap(Wrapped(10));
  TestSequence other_value = TestSequence::AddTo(this_value, 100);

  EXPECT_EQ(TestSequence::Difference(this_value, other_value), 100);
  EXPECT_EQ(TestSequence::Difference(other_value, this_value), 100);

  TestSequence minus_value = TestSequence::AddTo(this_value, -100);

  EXPECT_EQ(TestSequence::Difference(this_value, minus_value), 100);
  EXPECT_EQ(TestSequence::Difference(minus_value, this_value), 100);
}

}  // namespace
}  // namespace dcsctp
