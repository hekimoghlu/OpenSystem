/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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
#ifndef P2P_BASE_BASIC_PACKET_SOCKET_FACTORY_H_
#define P2P_BASE_BASIC_PACKET_SOCKET_FACTORY_H_

#include <stdint.h>

#include <memory>
#include <string>

#include "api/async_dns_resolver.h"
#include "api/packet_socket_factory.h"
#include "rtc_base/async_packet_socket.h"
#include "rtc_base/socket.h"
#include "rtc_base/socket_address.h"
#include "rtc_base/socket_factory.h"
#include "rtc_base/system/rtc_export.h"

namespace rtc {

class SocketFactory;

class RTC_EXPORT BasicPacketSocketFactory : public PacketSocketFactory {
 public:
  explicit BasicPacketSocketFactory(SocketFactory* socket_factory);
  ~BasicPacketSocketFactory() override;

  AsyncPacketSocket* CreateUdpSocket(const SocketAddress& local_address,
                                     uint16_t min_port,
                                     uint16_t max_port) override;
  AsyncListenSocket* CreateServerTcpSocket(const SocketAddress& local_address,
                                           uint16_t min_port,
                                           uint16_t max_port,
                                           int opts) override;
  AsyncPacketSocket* CreateClientTcpSocket(
      const SocketAddress& local_address,
      const SocketAddress& remote_address,
      const PacketSocketTcpOptions& tcp_options) override;

  std::unique_ptr<webrtc::AsyncDnsResolverInterface> CreateAsyncDnsResolver()
      override;

 private:
  int BindSocket(Socket* socket,
                 const SocketAddress& local_address,
                 uint16_t min_port,
                 uint16_t max_port);

  SocketFactory* socket_factory_;
};

}  // namespace rtc

#endif  // P2P_BASE_BASIC_PACKET_SOCKET_FACTORY_H_
