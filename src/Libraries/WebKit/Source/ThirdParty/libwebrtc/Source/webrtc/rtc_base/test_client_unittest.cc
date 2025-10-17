/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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
#include "rtc_base/test_client.h"

#include <utility>

#include "absl/memory/memory.h"
#include "rtc_base/async_tcp_socket.h"
#include "rtc_base/async_udp_socket.h"
#include "rtc_base/logging.h"
#include "rtc_base/net_helpers.h"
#include "rtc_base/net_test_helpers.h"
#include "rtc_base/physical_socket_server.h"
#include "rtc_base/socket.h"
#include "rtc_base/test_echo_server.h"
#include "rtc_base/thread.h"
#include "test/gtest.h"

namespace rtc {
namespace {

#define MAYBE_SKIP_IPV4                        \
  if (!HasIPv4Enabled()) {                     \
    RTC_LOG(LS_INFO) << "No IPv4... skipping"; \
    return;                                    \
  }

#define MAYBE_SKIP_IPV6                        \
  if (!HasIPv6Enabled()) {                     \
    RTC_LOG(LS_INFO) << "No IPv6... skipping"; \
    return;                                    \
  }

void TestUdpInternal(const SocketAddress& loopback) {
  rtc::PhysicalSocketServer socket_server;
  rtc::AutoSocketServerThread main_thread(&socket_server);
  Socket* socket = socket_server.CreateSocket(loopback.family(), SOCK_DGRAM);
  socket->Bind(loopback);

  TestClient client(std::make_unique<AsyncUDPSocket>(socket));
  SocketAddress addr = client.address(), from;
  EXPECT_EQ(3, client.SendTo("foo", 3, addr));
  EXPECT_TRUE(client.CheckNextPacket("foo", 3, &from));
  EXPECT_EQ(from, addr);
  EXPECT_TRUE(client.CheckNoPacket());
}

void TestTcpInternal(const SocketAddress& loopback) {
  rtc::PhysicalSocketServer socket_server;
  rtc::AutoSocketServerThread main_thread(&socket_server);
  TestEchoServer server(&main_thread, loopback);

  Socket* socket = socket_server.CreateSocket(loopback.family(), SOCK_STREAM);
  std::unique_ptr<AsyncTCPSocket> tcp_socket = absl::WrapUnique(
      AsyncTCPSocket::Create(socket, loopback, server.address()));
  ASSERT_TRUE(tcp_socket != nullptr);

  TestClient client(std::move(tcp_socket));
  SocketAddress addr = client.address(), from;
  EXPECT_TRUE(client.CheckConnected());
  EXPECT_EQ(3, client.Send("foo", 3));
  EXPECT_TRUE(client.CheckNextPacket("foo", 3, &from));
  EXPECT_EQ(from, server.address());
  EXPECT_TRUE(client.CheckNoPacket());
}

// Tests whether the TestClient can send UDP to itself.
TEST(TestClientTest, TestUdpIPv4) {
  MAYBE_SKIP_IPV4;
  TestUdpInternal(SocketAddress("127.0.0.1", 0));
}

#if defined(WEBRTC_LINUX)
#define MAYBE_TestUdpIPv6 DISABLED_TestUdpIPv6
#else
#define MAYBE_TestUdpIPv6 TestUdpIPv6
#endif
TEST(TestClientTest, MAYBE_TestUdpIPv6) {
  MAYBE_SKIP_IPV6;
  TestUdpInternal(SocketAddress("::1", 0));
}

// Tests whether the TestClient can connect to a server and exchange data.
TEST(TestClientTest, TestTcpIPv4) {
  MAYBE_SKIP_IPV4;
  TestTcpInternal(SocketAddress("127.0.0.1", 0));
}

#if defined(WEBRTC_LINUX)
#define MAYBE_TestTcpIPv6 DISABLED_TestTcpIPv6
#else
#define MAYBE_TestTcpIPv6 TestTcpIPv6
#endif
TEST(TestClientTest, MAYBE_TestTcpIPv6) {
  MAYBE_SKIP_IPV6;
  TestTcpInternal(SocketAddress("::1", 0));
}

}  // namespace
}  // namespace rtc
