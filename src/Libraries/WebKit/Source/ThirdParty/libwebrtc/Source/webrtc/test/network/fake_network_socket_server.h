/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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
#ifndef TEST_NETWORK_FAKE_NETWORK_SOCKET_SERVER_H_
#define TEST_NETWORK_FAKE_NETWORK_SOCKET_SERVER_H_

#include <set>
#include <vector>

#include "api/units/timestamp.h"
#include "rtc_base/event.h"
#include "rtc_base/socket.h"
#include "rtc_base/socket_server.h"
#include "rtc_base/synchronization/mutex.h"
#include "system_wrappers/include/clock.h"
#include "test/network/network_emulation.h"

namespace webrtc {
namespace test {
class FakeNetworkSocket;

// FakeNetworkSocketServer must outlive any sockets it creates.
class FakeNetworkSocketServer : public rtc::SocketServer {
 public:
  explicit FakeNetworkSocketServer(EndpointsContainer* endpoints_controller);
  ~FakeNetworkSocketServer() override;

  // rtc::SocketFactory methods:
  rtc::Socket* CreateSocket(int family, int type) override;

  // rtc::SocketServer methods:
  // Called by the network thread when this server is installed, kicking off the
  // message handler loop.
  void SetMessageQueue(rtc::Thread* thread) override;
  bool Wait(webrtc::TimeDelta max_wait_duration, bool process_io) override;
  void WakeUp() override;

 protected:
  friend class FakeNetworkSocket;
  EmulatedEndpointImpl* GetEndpointNode(const rtc::IPAddress& ip);
  void Unregister(FakeNetworkSocket* socket);

 private:
  const EndpointsContainer* endpoints_container_;
  rtc::Event wakeup_;
  rtc::Thread* thread_ = nullptr;

  Mutex lock_;
  std::vector<FakeNetworkSocket*> sockets_ RTC_GUARDED_BY(lock_);
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_NETWORK_FAKE_NETWORK_SOCKET_SERVER_H_
