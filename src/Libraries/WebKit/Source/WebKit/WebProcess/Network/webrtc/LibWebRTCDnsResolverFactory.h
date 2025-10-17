/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
#pragma once

#if USE(LIBWEBRTC)

#include <wtf/Compiler.h>

#include <WebCore/LibWebRTCMacros.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <webrtc/api/async_dns_resolver.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class LibWebRTCDnsResolverFactory final : public webrtc::AsyncDnsResolverFactoryInterface {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCDnsResolverFactory);
public:
    class Resolver : public webrtc::AsyncDnsResolverInterface {
    public:
        virtual void start(const rtc::SocketAddress&, Function<void()>&&) = 0;

    private:
        // webrtc::AsyncDnsResolverInterface
        void Start(const rtc::SocketAddress&, absl::AnyInvocable<void()>) final;
        void Start(const rtc::SocketAddress&, int family, absl::AnyInvocable<void()>) final;
    };

private:
    std::unique_ptr<webrtc::AsyncDnsResolverInterface> CreateAndResolve(const rtc::SocketAddress&, absl::AnyInvocable<void()>) final;
    std::unique_ptr<webrtc::AsyncDnsResolverInterface> CreateAndResolve(const rtc::SocketAddress&, int family, absl::AnyInvocable<void()>) final;
    std::unique_ptr<webrtc::AsyncDnsResolverInterface> Create() final;
};

} // namespace WebKit

#endif // USE(LIBWEBRTC)
