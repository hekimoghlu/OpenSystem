/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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
#include "p2p/base/turn_server.h"

#include <memory>

#include "p2p/base/basic_packet_socket_factory.h"
#include "p2p/base/port_interface.h"
#include "rtc_base/async_packet_socket.h"
#include "rtc_base/socket_address.h"
#include "rtc_base/thread.h"
#include "rtc_base/virtual_socket_server.h"
#include "test/gtest.h"

// NOTE: This is a work in progress. Currently this file only has tests for
// TurnServerConnection, a primitive class used by TurnServer.

namespace cricket {

class TurnServerConnectionTest : public ::testing::Test {
 public:
  TurnServerConnectionTest() : thread_(&vss_), socket_factory_(&vss_) {}

  void ExpectEqual(const TurnServerConnection& a,
                   const TurnServerConnection& b) {
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(b < a);
  }

  void ExpectNotEqual(const TurnServerConnection& a,
                      const TurnServerConnection& b) {
    EXPECT_FALSE(a == b);
    // We don't care which is less than the other, as long as only one is less
    // than the other.
    EXPECT_TRUE((a < b) != (b < a));
  }

 protected:
  rtc::VirtualSocketServer vss_;
  rtc::AutoSocketServerThread thread_;
  rtc::BasicPacketSocketFactory socket_factory_;
};

TEST_F(TurnServerConnectionTest, ComparisonOperators) {
  std::unique_ptr<rtc::AsyncPacketSocket> socket1(
      socket_factory_.CreateUdpSocket(rtc::SocketAddress("1.1.1.1", 1), 0, 0));
  std::unique_ptr<rtc::AsyncPacketSocket> socket2(
      socket_factory_.CreateUdpSocket(rtc::SocketAddress("2.2.2.2", 2), 0, 0));
  TurnServerConnection connection1(socket2->GetLocalAddress(), PROTO_UDP,
                                   socket1.get());
  TurnServerConnection connection2(socket2->GetLocalAddress(), PROTO_UDP,
                                   socket1.get());
  TurnServerConnection connection3(socket1->GetLocalAddress(), PROTO_UDP,
                                   socket2.get());
  TurnServerConnection connection4(socket2->GetLocalAddress(), PROTO_TCP,
                                   socket1.get());
  ExpectEqual(connection1, connection2);
  ExpectNotEqual(connection1, connection3);
  ExpectNotEqual(connection1, connection4);
}

}  // namespace cricket
