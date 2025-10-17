/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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
#include "pc/ice_transport.h"

#include <memory>
#include <utility>

#include "api/ice_transport_factory.h"
#include "api/make_ref_counted.h"
#include "api/scoped_refptr.h"
#include "p2p/base/fake_ice_transport.h"
#include "p2p/base/fake_port_allocator.h"
#include "rtc_base/internal/default_socket_server.h"
#include "test/gtest.h"
#include "test/scoped_key_value_config.h"

namespace webrtc {

class IceTransportTest : public ::testing::Test {
 protected:
  IceTransportTest()
      : socket_server_(rtc::CreateDefaultSocketServer()),
        main_thread_(socket_server_.get()) {}

  rtc::SocketServer* socket_server() const { return socket_server_.get(); }

  test::ScopedKeyValueConfig field_trials_;

 private:
  std::unique_ptr<rtc::SocketServer> socket_server_;
  rtc::AutoSocketServerThread main_thread_;
};

TEST_F(IceTransportTest, CreateNonSelfDeletingTransport) {
  auto cricket_transport =
      std::make_unique<cricket::FakeIceTransport>("name", 0, nullptr);
  auto ice_transport =
      rtc::make_ref_counted<IceTransportWithPointer>(cricket_transport.get());
  EXPECT_EQ(ice_transport->internal(), cricket_transport.get());
  ice_transport->Clear();
  EXPECT_NE(ice_transport->internal(), cricket_transport.get());
}

TEST_F(IceTransportTest, CreateSelfDeletingTransport) {
  std::unique_ptr<cricket::FakePortAllocator> port_allocator(
      std::make_unique<cricket::FakePortAllocator>(
          nullptr,
          std::make_unique<rtc::BasicPacketSocketFactory>(socket_server()),
          &field_trials_));
  IceTransportInit init;
  init.set_port_allocator(port_allocator.get());
  auto ice_transport = CreateIceTransport(std::move(init));
  EXPECT_NE(nullptr, ice_transport->internal());
}

}  // namespace webrtc
