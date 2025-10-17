/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#ifndef P2P_BASE_TEST_STUN_SERVER_H_
#define P2P_BASE_TEST_STUN_SERVER_H_

#include <memory>

#include "api/transport/stun.h"
#include "p2p/base/stun_server.h"
#include "rtc_base/async_udp_socket.h"
#include "rtc_base/socket_address.h"
#include "rtc_base/socket_server.h"
#include "rtc_base/thread.h"

namespace cricket {

// A test STUN server. Useful for unit tests.
class TestStunServer : StunServer {
 public:
  using StunServerPtr =
      std::unique_ptr<TestStunServer, std::function<void(TestStunServer*)>>;
  static StunServerPtr Create(rtc::SocketServer* ss,
                              const rtc::SocketAddress& addr,
                              rtc::Thread& network_thread);

  // Set a fake STUN address to return to the client.
  void set_fake_stun_addr(const rtc::SocketAddress& addr) {
    fake_stun_addr_ = addr;
  }

 private:
  static void DeleteOnNetworkThread(TestStunServer* server);

  TestStunServer(rtc::AsyncUDPSocket* socket, rtc::Thread& network_thread)
      : StunServer(socket), network_thread_(network_thread) {}

  void OnBindingRequest(StunMessage* msg,
                        const rtc::SocketAddress& remote_addr) override;

 private:
  rtc::SocketAddress fake_stun_addr_;
  rtc::Thread& network_thread_;
};

}  // namespace cricket

#endif  // P2P_BASE_TEST_STUN_SERVER_H_
