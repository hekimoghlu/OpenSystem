/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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

#include "LegacyCustomProtocolID.h"
#include "MessageReceiver.h"
#include <wtf/CheckedRef.h>
#include <wtf/WeakPtr.h>

#if PLATFORM(COCOA)
#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>
OBJC_CLASS WKCustomProtocolLoader;
#endif

namespace WebCore {
class ResourceError;
class ResourceRequest;
class ResourceResponse;
} // namespace WebCore

namespace WebKit {

enum class CacheStoragePolicy : uint8_t;
class NetworkProcessProxy;

class LegacyCustomProtocolManagerProxy final : public IPC::MessageReceiver {
public:
    explicit LegacyCustomProtocolManagerProxy(NetworkProcessProxy&);
    ~LegacyCustomProtocolManagerProxy();

    void startLoading(LegacyCustomProtocolID, const WebCore::ResourceRequest&);
    void stopLoading(LegacyCustomProtocolID);

    void invalidate();

    void wasRedirectedToRequest(LegacyCustomProtocolID, const WebCore::ResourceRequest&, const WebCore::ResourceResponse&);
    void didReceiveResponse(LegacyCustomProtocolID, const WebCore::ResourceResponse&, CacheStoragePolicy);
    void didLoadData(LegacyCustomProtocolID, std::span<const uint8_t>);
    void didFailWithError(LegacyCustomProtocolID, const WebCore::ResourceError&);
    void didFinishLoading(LegacyCustomProtocolID);

    void ref() const final;
    void deref() const final;

private:
    Ref<NetworkProcessProxy> protectedProcess();

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    WeakRef<NetworkProcessProxy> m_networkProcessProxy;
};

} // namespace WebKit
