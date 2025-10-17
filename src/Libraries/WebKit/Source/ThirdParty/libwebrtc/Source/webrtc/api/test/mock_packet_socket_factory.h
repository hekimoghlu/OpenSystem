/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#ifndef API_TEST_MOCK_PACKET_SOCKET_FACTORY_H_
#define API_TEST_MOCK_PACKET_SOCKET_FACTORY_H_

#include <cstdint>
#include <memory>
#include <type_traits>

#include "api/async_dns_resolver.h"
#include "api/packet_socket_factory.h"
#include "rtc_base/async_packet_socket.h"
#include "rtc_base/socket_address.h"
#include "test/gmock.h"

namespace rtc {
class MockPacketSocketFactory : public PacketSocketFactory {
 public:
  MOCK_METHOD(AsyncPacketSocket*,
              CreateUdpSocket,
              (const SocketAddress&, uint16_t, uint16_t),
              (override));
  MOCK_METHOD(AsyncListenSocket*,
              CreateServerTcpSocket,
              (const SocketAddress&, uint16_t, uint16_t, int opts),
              (override));
  MOCK_METHOD(AsyncPacketSocket*,
              CreateClientTcpSocket,
              (const SocketAddress& local_address,
               const SocketAddress&,
               const PacketSocketTcpOptions&),
              (override));
  MOCK_METHOD(std::unique_ptr<webrtc::AsyncDnsResolverInterface>,
              CreateAsyncDnsResolver,
              (),
              (override));
};

static_assert(!std::is_abstract_v<MockPacketSocketFactory>, "");

}  // namespace rtc

#endif  // API_TEST_MOCK_PACKET_SOCKET_FACTORY_H_
