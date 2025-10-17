/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
#ifndef P2P_BASE_MOCK_DNS_RESOLVING_PACKET_SOCKET_FACTORY_H_
#define P2P_BASE_MOCK_DNS_RESOLVING_PACKET_SOCKET_FACTORY_H_

#include <functional>
#include <memory>

#include "api/test/mock_async_dns_resolver.h"
#include "p2p/base/basic_packet_socket_factory.h"

namespace rtc {

// A PacketSocketFactory implementation for tests that uses a mock DnsResolver
// and allows setting expectations on the resolver and results.
class MockDnsResolvingPacketSocketFactory : public BasicPacketSocketFactory {
 public:
  using Expectations = std::function<void(webrtc::MockAsyncDnsResolver*,
                                          webrtc::MockAsyncDnsResolverResult*)>;

  explicit MockDnsResolvingPacketSocketFactory(SocketFactory* socket_factory)
      : BasicPacketSocketFactory(socket_factory) {}

  std::unique_ptr<webrtc::AsyncDnsResolverInterface> CreateAsyncDnsResolver()
      override {
    std::unique_ptr<webrtc::MockAsyncDnsResolver> resolver =
        std::make_unique<webrtc::MockAsyncDnsResolver>();
    if (expectations_) {
      expectations_(resolver.get(), &resolver_result_);
    }
    return resolver;
  }

  void SetExpectations(Expectations expectations) {
    expectations_ = expectations;
  }

 private:
  webrtc::MockAsyncDnsResolverResult resolver_result_;
  Expectations expectations_;
};

}  // namespace rtc

#endif  // P2P_BASE_MOCK_DNS_RESOLVING_PACKET_SOCKET_FACTORY_H_
