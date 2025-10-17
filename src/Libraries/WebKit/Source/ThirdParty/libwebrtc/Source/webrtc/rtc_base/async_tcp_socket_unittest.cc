/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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
#include "rtc_base/async_tcp_socket.h"

#include <memory>
#include <string>

#include "rtc_base/gunit.h"
#include "rtc_base/virtual_socket_server.h"

namespace rtc {

class AsyncTCPSocketTest : public ::testing::Test, public sigslot::has_slots<> {
 public:
  AsyncTCPSocketTest()
      : vss_(new rtc::VirtualSocketServer()),
        socket_(vss_->CreateSocket(SOCK_STREAM)),
        tcp_socket_(new AsyncTCPSocket(socket_, true)),
        ready_to_send_(false) {
    tcp_socket_->SignalReadyToSend.connect(this,
                                           &AsyncTCPSocketTest::OnReadyToSend);
  }

  void OnReadyToSend(rtc::AsyncPacketSocket* socket) { ready_to_send_ = true; }

 protected:
  std::unique_ptr<VirtualSocketServer> vss_;
  Socket* socket_;
  std::unique_ptr<AsyncTCPSocket> tcp_socket_;
  bool ready_to_send_;
};

TEST_F(AsyncTCPSocketTest, OnWriteEvent) {
  EXPECT_FALSE(ready_to_send_);
  socket_->SignalWriteEvent(socket_);
  EXPECT_TRUE(ready_to_send_);
}

}  // namespace rtc
