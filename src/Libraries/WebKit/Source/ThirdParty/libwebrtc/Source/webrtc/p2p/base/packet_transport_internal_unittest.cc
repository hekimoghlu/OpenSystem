/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
#include "p2p/base/packet_transport_internal.h"

#include "p2p/base/fake_packet_transport.h"
#include "rtc_base/gunit.h"
#include "rtc_base/network/received_packet.h"
#include "rtc_base/third_party/sigslot/sigslot.h"
#include "test/gmock.h"

namespace {

using ::testing::MockFunction;

TEST(PacketTransportInternal,
     NotifyPacketReceivedPassthrougPacketToRegisteredListener) {
  rtc::FakePacketTransport packet_transport("test");
  MockFunction<void(rtc::PacketTransportInternal*, const rtc::ReceivedPacket&)>
      receiver;

  packet_transport.RegisterReceivedPacketCallback(&receiver,
                                                  receiver.AsStdFunction());
  EXPECT_CALL(receiver, Call)
      .WillOnce(
          [](rtc::PacketTransportInternal*, const rtc::ReceivedPacket& packet) {
            EXPECT_EQ(packet.decryption_info(),
                      rtc::ReceivedPacket::kDtlsDecrypted);
          });
  packet_transport.NotifyPacketReceived(rtc::ReceivedPacket(
      {}, rtc::SocketAddress(), std::nullopt, rtc::EcnMarking::kNotEct,
      rtc::ReceivedPacket::kDtlsDecrypted));

  packet_transport.DeregisterReceivedPacketCallback(&receiver);
}

TEST(PacketTransportInternal, NotifiesOnceOnClose) {
  rtc::FakePacketTransport packet_transport("test");
  int call_count = 0;
  packet_transport.SetOnCloseCallback([&]() { ++call_count; });
  ASSERT_EQ(call_count, 0);
  packet_transport.NotifyOnClose();
  EXPECT_EQ(call_count, 1);
  packet_transport.NotifyOnClose();
  EXPECT_EQ(call_count, 1);  // Call count should not have increased.
}

}  // namespace
