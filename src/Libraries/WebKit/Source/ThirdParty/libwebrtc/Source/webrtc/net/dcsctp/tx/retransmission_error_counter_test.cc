/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#include "net/dcsctp/tx/retransmission_error_counter.h"

#include "net/dcsctp/public/dcsctp_options.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {

TEST(RetransmissionErrorCounterTest, HasInitialValue) {
  DcSctpOptions options;
  RetransmissionErrorCounter counter("log: ", options);
  EXPECT_EQ(counter.value(), 0);
}

TEST(RetransmissionErrorCounterTest, ReturnsFalseAtMaximumValue) {
  DcSctpOptions options;
  options.max_retransmissions = 5;
  RetransmissionErrorCounter counter("log: ", options);
  EXPECT_TRUE(counter.Increment("test"));   // 1
  EXPECT_TRUE(counter.Increment("test"));   // 2
  EXPECT_TRUE(counter.Increment("test"));   // 3
  EXPECT_TRUE(counter.Increment("test"));   // 4
  EXPECT_TRUE(counter.Increment("test"));   // 5
  EXPECT_FALSE(counter.Increment("test"));  // Too many retransmissions
}

TEST(RetransmissionErrorCounterTest, CanHandleZeroRetransmission) {
  DcSctpOptions options;
  options.max_retransmissions = 0;
  RetransmissionErrorCounter counter("log: ", options);
  EXPECT_FALSE(counter.Increment("test"));  // One is too many.
}

TEST(RetransmissionErrorCounterTest, IsExhaustedAtMaximum) {
  DcSctpOptions options;
  options.max_retransmissions = 3;
  RetransmissionErrorCounter counter("log: ", options);
  EXPECT_TRUE(counter.Increment("test"));  // 1
  EXPECT_FALSE(counter.IsExhausted());
  EXPECT_TRUE(counter.Increment("test"));  // 2
  EXPECT_FALSE(counter.IsExhausted());
  EXPECT_TRUE(counter.Increment("test"));  // 3
  EXPECT_FALSE(counter.IsExhausted());
  EXPECT_FALSE(counter.Increment("test"));  // Too many retransmissions
  EXPECT_TRUE(counter.IsExhausted());
  EXPECT_FALSE(counter.Increment("test"));  // One after too many
  EXPECT_TRUE(counter.IsExhausted());
}

TEST(RetransmissionErrorCounterTest, ClearingCounter) {
  DcSctpOptions options;
  options.max_retransmissions = 3;
  RetransmissionErrorCounter counter("log: ", options);
  EXPECT_TRUE(counter.Increment("test"));  // 1
  EXPECT_TRUE(counter.Increment("test"));  // 2
  counter.Clear();
  EXPECT_TRUE(counter.Increment("test"));  // 1
  EXPECT_TRUE(counter.Increment("test"));  // 2
  EXPECT_TRUE(counter.Increment("test"));  // 3
  EXPECT_FALSE(counter.IsExhausted());
  EXPECT_FALSE(counter.Increment("test"));  // Too many retransmissions
  EXPECT_TRUE(counter.IsExhausted());
}

TEST(RetransmissionErrorCounterTest, CanBeLimitless) {
  DcSctpOptions options;
  options.max_retransmissions = std::nullopt;
  RetransmissionErrorCounter counter("log: ", options);
  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(counter.Increment("test"));
    EXPECT_FALSE(counter.IsExhausted());
  }
}

}  // namespace
}  // namespace dcsctp
