/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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
#ifndef API_PACKET_SOCKET_FACTORY_H_
#define API_PACKET_SOCKET_FACTORY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "api/async_dns_resolver.h"
#include "rtc_base/async_packet_socket.h"
#include "rtc_base/socket_address.h"
#include "rtc_base/system/rtc_export.h"

namespace rtc {

class SSLCertificateVerifier;
class AsyncResolverInterface;

struct PacketSocketTcpOptions {
  PacketSocketTcpOptions() = default;
  ~PacketSocketTcpOptions() = default;

  int opts = 0;
  std::vector<std::string> tls_alpn_protocols;
  std::vector<std::string> tls_elliptic_curves;
  // An optional custom SSL certificate verifier that an API user can provide to
  // inject their own certificate verification logic (not available to users
  // outside of the WebRTC repo).
  SSLCertificateVerifier* tls_cert_verifier = nullptr;
};

class RTC_EXPORT PacketSocketFactory {
 public:
  enum Options {
    OPT_STUN = 0x04,

    // The TLS options below are mutually exclusive.
    OPT_TLS = 0x02,           // Real and secure TLS.
    OPT_TLS_FAKE = 0x01,      // Fake TLS with a dummy SSL handshake.
    OPT_TLS_INSECURE = 0x08,  // Insecure TLS without certificate validation.

    // Deprecated, use OPT_TLS_FAKE.
    OPT_SSLTCP = OPT_TLS_FAKE,
  };

  PacketSocketFactory() = default;
  virtual ~PacketSocketFactory() = default;

  virtual AsyncPacketSocket* CreateUdpSocket(const SocketAddress& address,
                                             uint16_t min_port,
                                             uint16_t max_port) = 0;
  virtual AsyncListenSocket* CreateServerTcpSocket(
      const SocketAddress& local_address,
      uint16_t min_port,
      uint16_t max_port,
      int opts) = 0;

  virtual AsyncPacketSocket* CreateClientTcpSocket(
      const SocketAddress& local_address,
      const SocketAddress& remote_address,
      const PacketSocketTcpOptions& tcp_options) = 0;

  virtual std::unique_ptr<webrtc::AsyncDnsResolverInterface>
  CreateAsyncDnsResolver() = 0;

 private:
  PacketSocketFactory(const PacketSocketFactory&) = delete;
  PacketSocketFactory& operator=(const PacketSocketFactory&) = delete;
};

}  // namespace rtc

#endif  // API_PACKET_SOCKET_FACTORY_H_
