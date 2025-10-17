/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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
#include "rtc_base/async_packet_socket.h"

#include "rtc_base/socket_address.h"
#include "rtc_base/third_party/sigslot/sigslot.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace rtc {
namespace {

using ::testing::MockFunction;

class MockAsyncPacketSocket : public rtc::AsyncPacketSocket {
 public:
  ~MockAsyncPacketSocket() = default;

  MOCK_METHOD(SocketAddress, GetLocalAddress, (), (const, override));
  MOCK_METHOD(SocketAddress, GetRemoteAddress, (), (const, override));
  MOCK_METHOD(int,
              Send,
              (const void* pv, size_t cb, const rtc::PacketOptions& options),
              (override));

  MOCK_METHOD(int,
              SendTo,
              (const void* pv,
               size_t cb,
               const SocketAddress& addr,
               const rtc::PacketOptions& options),
              (override));
  MOCK_METHOD(int, Close, (), (override));
  MOCK_METHOD(State, GetState, (), (const, override));
  MOCK_METHOD(int,
              GetOption,
              (rtc::Socket::Option opt, int* value),
              (override));
  MOCK_METHOD(int, SetOption, (rtc::Socket::Option opt, int value), (override));
  MOCK_METHOD(int, GetError, (), (const, override));
  MOCK_METHOD(void, SetError, (int error), (override));

  using AsyncPacketSocket::NotifyPacketReceived;
};

TEST(AsyncPacketSocket, RegisteredCallbackReceivePacketsFromNotify) {
  MockAsyncPacketSocket mock_socket;
  MockFunction<void(AsyncPacketSocket*, const rtc::ReceivedPacket&)>
      received_packet;

  EXPECT_CALL(received_packet, Call);
  mock_socket.RegisterReceivedPacketCallback(received_packet.AsStdFunction());
  mock_socket.NotifyPacketReceived(ReceivedPacket({}, SocketAddress()));
}

}  // namespace
}  // namespace rtc
