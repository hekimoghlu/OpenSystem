/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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

#include "WebPageProxyIdentifier.h"
#include <WebCore/EmptyFrameLoaderClient.h>
#include <WebCore/ScriptExecutionContextIdentifier.h>

namespace WebCore {

class FrameLoader;

}

namespace WebKit {

class RemoteWorkerFrameLoaderClient final : public WebCore::EmptyFrameLoaderClient {
public:
    RemoteWorkerFrameLoaderClient(WebCore::FrameLoader&, WebPageProxyIdentifier, WebCore::PageIdentifier, const String& userAgent);

    WebPageProxyIdentifier webPageProxyID() const { return m_webPageProxyID; }

    void setUserAgent(String&& userAgent) { m_userAgent = WTFMove(userAgent); }

    void setServiceWorkerPageIdentifier(WebCore::ScriptExecutionContextIdentifier serviceWorkerPageIdentifier) { m_serviceWorkerPageIdentifier = serviceWorkerPageIdentifier; }
    std::optional<WebCore::ScriptExecutionContextIdentifier> serviceWorkerPageIdentifier() const { return m_serviceWorkerPageIdentifier; }

private:
    Ref<WebCore::DocumentLoader> createDocumentLoader(const WebCore::ResourceRequest&, const WebCore::SubstituteData&) final;

    bool shouldUseCredentialStorage(WebCore::DocumentLoader*, WebCore::ResourceLoaderIdentifier) final { return true; }
    bool isRemoteWorkerFrameLoaderClient() const final { return true; }

    String userAgent(const URL&) const final { return m_userAgent; }

    WebPageProxyIdentifier m_webPageProxyID;
    String m_userAgent;
    std::optional<WebCore::ScriptExecutionContextIdentifier> m_serviceWorkerPageIdentifier;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::RemoteWorkerFrameLoaderClient)
    static bool isType(const WebCore::LocalFrameLoaderClient& frameLoaderClient) { return frameLoaderClient.isRemoteWorkerFrameLoaderClient(); }
SPECIALIZE_TYPE_TRAITS_END()
