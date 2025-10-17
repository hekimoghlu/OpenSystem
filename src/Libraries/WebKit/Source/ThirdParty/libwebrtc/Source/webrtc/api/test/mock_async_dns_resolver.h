/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#ifndef API_TEST_MOCK_ASYNC_DNS_RESOLVER_H_
#define API_TEST_MOCK_ASYNC_DNS_RESOLVER_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "api/async_dns_resolver.h"
#include "rtc_base/socket_address.h"
#include "test/gmock.h"

namespace webrtc {

class MockAsyncDnsResolverResult : public AsyncDnsResolverResult {
 public:
  MOCK_METHOD(bool,
              GetResolvedAddress,
              (int, rtc::SocketAddress*),
              (const, override));
  MOCK_METHOD(int, GetError, (), (const, override));
};

class MockAsyncDnsResolver : public AsyncDnsResolverInterface {
 public:
  MOCK_METHOD(void,
              Start,
              (const rtc::SocketAddress&, absl::AnyInvocable<void()>),
              (override));
  MOCK_METHOD(void,
              Start,
              (const rtc::SocketAddress&,
               int family,
               absl::AnyInvocable<void()>),
              (override));
  MOCK_METHOD(AsyncDnsResolverResult&, result, (), (const, override));
};

class MockAsyncDnsResolverFactory : public AsyncDnsResolverFactoryInterface {
 public:
  MOCK_METHOD(std::unique_ptr<webrtc::AsyncDnsResolverInterface>,
              CreateAndResolve,
              (const rtc::SocketAddress&, absl::AnyInvocable<void()>),
              (override));
  MOCK_METHOD(std::unique_ptr<webrtc::AsyncDnsResolverInterface>,
              CreateAndResolve,
              (const rtc::SocketAddress&, int, absl::AnyInvocable<void()>),
              (override));
  MOCK_METHOD(std::unique_ptr<webrtc::AsyncDnsResolverInterface>,
              Create,
              (),
              (override));
};

}  // namespace webrtc

#endif  // API_TEST_MOCK_ASYNC_DNS_RESOLVER_H_
