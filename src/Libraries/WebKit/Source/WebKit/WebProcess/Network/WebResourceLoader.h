/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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

#include "Connection.h"
#include "MessageSender.h"
#include "WebPageProxyIdentifier.h"
#include "WebResourceInterceptController.h"
#include <WebCore/FrameIdentifier.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/ResourceLoaderIdentifier.h>
#include <WebCore/ShareableResource.h>
#include <wtf/MonotonicTime.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace IPC {
class FormDataReference;
class SharedBufferReference;
}

namespace WebCore {
class ContentFilterUnblockHandler;
class NetworkLoadMetrics;
class ResourceError;
class ResourceLoader;
class ResourceRequest;
class ResourceResponse;
class SubstituteData;
enum class MainFrameMainResource : bool;
}

namespace WebKit {

enum class PrivateRelayed : bool;

class WebResourceLoader : public RefCounted<WebResourceLoader>, public IPC::MessageSender {
public:
    struct TrackingParameters {
        WebPageProxyIdentifier webPageProxyID;
        WebCore::PageIdentifier pageID;
        WebCore::FrameIdentifier frameID;
        WebCore::ResourceLoaderIdentifier resourceID;
    };

    static Ref<WebResourceLoader> create(Ref<WebCore::ResourceLoader>&&, const std::optional<TrackingParameters>&);

    ~WebResourceLoader();

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    WebCore::ResourceLoader* resourceLoader() const { return m_coreLoader.get(); }

    void detachFromCoreLoader();

private:
    WebResourceLoader(Ref<WebCore::ResourceLoader>&&, const std::optional<TrackingParameters>&);

    // IPC::MessageSender
    IPC::Connection* messageSenderConnection() const override;
    uint64_t messageSenderDestinationID() const override;

    void willSendRequest(WebCore::ResourceRequest&&, IPC::FormDataReference&& requestBody, WebCore::ResourceResponse&&, CompletionHandler<void(WebCore::ResourceRequest&&, bool)>&&);
    void didSendData(uint64_t bytesSent, uint64_t totalBytesToBeSent);
    void didReceiveResponse(WebCore::ResourceResponse&&, PrivateRelayed, bool needsContinueDidReceiveResponseMessage, std::optional<WebCore::NetworkLoadMetrics>&&);
    void didReceiveData(IPC::SharedBufferReference&& data, uint64_t encodedDataLength);
    void didFinishResourceLoad(WebCore::NetworkLoadMetrics&&);
    void didFailResourceLoad(const WebCore::ResourceError&);
    void didFailServiceWorkerLoad(const WebCore::ResourceError&);
    void serviceWorkerDidNotHandle();
    void didBlockAuthenticationChallenge();
    void setWorkerStart(MonotonicTime value) { m_workerStart = value; }

    void stopLoadingAfterXFrameOptionsOrContentSecurityPolicyDenied(const WebCore::ResourceResponse&);

    WebCore::MainFrameMainResource mainFrameMainResource() const;
    
#if ENABLE(SHAREABLE_RESOURCE)
    void didReceiveResource(WebCore::ShareableResource::Handle&&);
#endif

#if ENABLE(CONTENT_FILTERING)
    void contentFilterDidBlockLoad(const WebCore::ContentFilterUnblockHandler&, String&& unblockRequestDeniedScript, const WebCore::ResourceError&, const URL& blockedPageURL, WebCore::SubstituteData&&);
#endif
    
    RefPtr<WebCore::ResourceLoader> m_coreLoader;
    const std::optional<TrackingParameters> m_trackingParameters;
    WebResourceInterceptController m_interceptController;
    size_t m_numBytesReceived { 0 };

#if ASSERT_ENABLED
    bool m_isProcessingNetworkResponse { false };
#endif

    Seconds timeSinceLoadStart() const { return MonotonicTime::now() - m_loadStart; }

    const MonotonicTime m_loadStart;
    MonotonicTime m_workerStart;
};

} // namespace WebKit
