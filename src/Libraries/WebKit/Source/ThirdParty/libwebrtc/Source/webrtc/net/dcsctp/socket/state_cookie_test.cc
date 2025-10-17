/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
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
#include "net/dcsctp/socket/state_cookie.h"

#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::SizeIs;

TEST(StateCookieTest, SerializeAndDeserialize) {
  Capabilities capabilities = {.partial_reliability = true,
                               .message_interleaving = false,
                               .reconfig = true,
                               .zero_checksum = true,
                               .negotiated_maximum_incoming_streams = 123,
                               .negotiated_maximum_outgoing_streams = 234};
  StateCookie cookie(/*peer_tag=*/VerificationTag(123),
                     /*my_tag=*/VerificationTag(321),
                     /*peer_initial_tsn=*/TSN(456), /*my_initial_tsn=*/TSN(654),
                     /*a_rwnd=*/789, TieTag(101112), capabilities);
  std::vector<uint8_t> serialized = cookie.Serialize();
  EXPECT_THAT(serialized, SizeIs(StateCookie::kCookieSize));
  ASSERT_HAS_VALUE_AND_ASSIGN(StateCookie deserialized,
                              StateCookie::Deserialize(serialized));
  EXPECT_EQ(deserialized.peer_tag(), VerificationTag(123));
  EXPECT_EQ(deserialized.my_tag(), VerificationTag(321));
  EXPECT_EQ(deserialized.peer_initial_tsn(), TSN(456));
  EXPECT_EQ(deserialized.my_initial_tsn(), TSN(654));
  EXPECT_EQ(deserialized.a_rwnd(), 789u);
  EXPECT_EQ(deserialized.tie_tag(), TieTag(101112));
  EXPECT_TRUE(deserialized.capabilities().partial_reliability);
  EXPECT_FALSE(deserialized.capabilities().message_interleaving);
  EXPECT_TRUE(deserialized.capabilities().reconfig);
  EXPECT_TRUE(deserialized.capabilities().zero_checksum);
  EXPECT_EQ(deserialized.capabilities().negotiated_maximum_incoming_streams,
            123);
  EXPECT_EQ(deserialized.capabilities().negotiated_maximum_outgoing_streams,
            234);
}

TEST(StateCookieTest, ValidateMagicValue) {
  Capabilities capabilities = {.partial_reliability = true,
                               .message_interleaving = false,
                               .reconfig = true};
  StateCookie cookie(/*peer_tag=*/VerificationTag(123),
                     /*my_tag=*/VerificationTag(321),
                     /*peer_initial_tsn=*/TSN(456), /*my_initial_tsn=*/TSN(654),
                     /*a_rwnd=*/789, TieTag(101112), capabilities);
  std::vector<uint8_t> serialized = cookie.Serialize();
  ASSERT_THAT(serialized, SizeIs(StateCookie::kCookieSize));

  absl::string_view magic(reinterpret_cast<const char*>(serialized.data()), 8);
  EXPECT_EQ(magic, "dcSCTP00");
}

}  // namespace
}  // namespace dcsctp
