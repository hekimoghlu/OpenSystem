/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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
#include "p2p/base/test_stun_server.h"

#include <memory>

#include "rtc_base/socket.h"
#include "rtc_base/socket_server.h"

namespace cricket {

std::unique_ptr<TestStunServer, std::function<void(TestStunServer*)>>
TestStunServer::Create(rtc::SocketServer* ss,
                       const rtc::SocketAddress& addr,
                       rtc::Thread& network_thread) {
  rtc::Socket* socket = ss->CreateSocket(addr.family(), SOCK_DGRAM);
  rtc::AsyncUDPSocket* udp_socket = rtc::AsyncUDPSocket::Create(socket, addr);
  TestStunServer* server = nullptr;
  network_thread.BlockingCall(
      [&]() { server = new TestStunServer(udp_socket, network_thread); });
  std::unique_ptr<TestStunServer, std::function<void(TestStunServer*)>> result(
      server, [&](TestStunServer* server) {
        network_thread.BlockingCall([server]() { delete server; });
      });
  return result;
}

void TestStunServer::OnBindingRequest(StunMessage* msg,
                                      const rtc::SocketAddress& remote_addr) {
  RTC_DCHECK_RUN_ON(&network_thread_);
  if (fake_stun_addr_.IsNil()) {
    StunServer::OnBindingRequest(msg, remote_addr);
  } else {
    StunMessage response(STUN_BINDING_RESPONSE, msg->transaction_id());
    GetStunBindResponse(msg, fake_stun_addr_, &response);
    SendResponse(response, remote_addr);
  }
}

}  // namespace cricket
