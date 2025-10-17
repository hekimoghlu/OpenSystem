/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#include "rtc_base/async_udp_socket.h"

#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"
#include "rtc_base/async_packet_socket.h"
#include "rtc_base/socket.h"
#include "rtc_base/socket_address.h"
#include "rtc_base/virtual_socket_server.h"
#include "test/gtest.h"

namespace rtc {

static const SocketAddress kAddr("22.22.22.22", 0);

TEST(AsyncUDPSocketTest, SetSocketOptionIfEctChange) {
  VirtualSocketServer socket_server;
  Socket* socket = socket_server.CreateSocket(kAddr.family(), SOCK_DGRAM);
  std::unique_ptr<AsyncUDPSocket> udp__socket =
      absl::WrapUnique(AsyncUDPSocket::Create(socket, kAddr));

  int ect = 0;
  socket->GetOption(Socket::OPT_SEND_ECN, &ect);
  ASSERT_EQ(ect, 0);

  uint8_t buffer[] = "hello";
  rtc::PacketOptions packet_options;
  packet_options.ecn_1 = false;
  udp__socket->SendTo(buffer, 5, kAddr, packet_options);
  socket->GetOption(Socket::OPT_SEND_ECN, &ect);
  EXPECT_EQ(ect, 0);

  packet_options.ecn_1 = true;
  udp__socket->SendTo(buffer, 5, kAddr, packet_options);
  socket->GetOption(Socket::OPT_SEND_ECN, &ect);
  EXPECT_EQ(ect, 1);

  packet_options.ecn_1 = false;
  udp__socket->SendTo(buffer, 5, kAddr, packet_options);
  socket->GetOption(Socket::OPT_SEND_ECN, &ect);
  EXPECT_EQ(ect, 0);
}

}  // namespace rtc
