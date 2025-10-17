/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#include "p2p/base/basic_async_resolver_factory.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "api/async_dns_resolver.h"
#include "rtc_base/async_dns_resolver.h"
#include "rtc_base/logging.h"

namespace webrtc {

std::unique_ptr<webrtc::AsyncDnsResolverInterface>
BasicAsyncDnsResolverFactory::Create() {
  return std::make_unique<AsyncDnsResolver>();
}

std::unique_ptr<webrtc::AsyncDnsResolverInterface>
BasicAsyncDnsResolverFactory::CreateAndResolve(
    const rtc::SocketAddress& addr,
    absl::AnyInvocable<void()> callback) {
  std::unique_ptr<webrtc::AsyncDnsResolverInterface> resolver = Create();
  resolver->Start(addr, std::move(callback));
  return resolver;
}

std::unique_ptr<webrtc::AsyncDnsResolverInterface>
BasicAsyncDnsResolverFactory::CreateAndResolve(
    const rtc::SocketAddress& addr,
    int family,
    absl::AnyInvocable<void()> callback) {
  std::unique_ptr<webrtc::AsyncDnsResolverInterface> resolver = Create();
  resolver->Start(addr, family, std::move(callback));
  return resolver;
}

}  // namespace webrtc
