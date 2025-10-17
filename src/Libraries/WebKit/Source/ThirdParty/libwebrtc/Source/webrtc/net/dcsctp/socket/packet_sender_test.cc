/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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
#include "net/dcsctp/socket/packet_sender.h"

#include "net/dcsctp/common/internal_types.h"
#include "net/dcsctp/packet/chunk/cookie_ack_chunk.h"
#include "net/dcsctp/socket/mock_dcsctp_socket_callbacks.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::_;

constexpr VerificationTag kVerificationTag(123);

class PacketSenderTest : public testing::Test {
 protected:
  PacketSenderTest() : sender_(callbacks_, on_send_fn_.AsStdFunction()) {}

  SctpPacket::Builder PacketBuilder() const {
    return SctpPacket::Builder(kVerificationTag, options_);
  }

  DcSctpOptions options_;
  testing::NiceMock<MockDcSctpSocketCallbacks> callbacks_;
  testing::MockFunction<void(rtc::ArrayView<const uint8_t>, SendPacketStatus)>
      on_send_fn_;
  PacketSender sender_;
};

TEST_F(PacketSenderTest, SendPacketCallsCallback) {
  EXPECT_CALL(on_send_fn_, Call(_, SendPacketStatus::kSuccess));
  EXPECT_TRUE(sender_.Send(PacketBuilder().Add(CookieAckChunk())));

  EXPECT_CALL(callbacks_, SendPacketWithStatus)
      .WillOnce(testing::Return(SendPacketStatus::kError));
  EXPECT_CALL(on_send_fn_, Call(_, SendPacketStatus::kError));
  EXPECT_FALSE(sender_.Send(PacketBuilder().Add(CookieAckChunk())));
}

}  // namespace
}  // namespace dcsctp
