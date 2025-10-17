/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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

#include "NetworkDataTask.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/FrameIdentifier.h>
#include <WebCore/NetworkLoadInformation.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/StoredCredentialsPolicy.h>
#include <pal/SessionID.h>
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class ResourceError;
class SecurityOrigin;
class SharedBuffer;
enum class AdvancedPrivacyProtections : uint16_t;
}

namespace WebKit {

class NetworkProcess;
class NetworkResourceLoader;

class NetworkCORSPreflightChecker final : public RefCounted<NetworkCORSPreflightChecker>, private NetworkDataTaskClient {
    WTF_MAKE_TZONE_ALLOCATED(NetworkCORSPreflightChecker);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    struct Parameters {
        WebCore::ResourceRequest originalRequest;
        Ref<WebCore::SecurityOrigin> sourceOrigin;
        Ref<WebCore::SecurityOrigin> topOrigin;
        String referrer;
        String userAgent;
        PAL::SessionID sessionID;
        Markable<WebPageProxyIdentifier> webPageProxyID;
        WebCore::StoredCredentialsPolicy storedCredentialsPolicy;
        bool allowPrivacyProxy { true };
        OptionSet<WebCore::AdvancedPrivacyProtections> advancedPrivacyProtections;
        bool includeFetchMetadata { false };
    };
    using CompletionCallback = CompletionHandler<void(WebCore::ResourceError&&)>;

    static Ref<NetworkCORSPreflightChecker> create(NetworkProcess& networkProcess, NetworkResourceLoader* networkResourceLoader, Parameters&& parameters, bool shouldCaptureExtraNetworkLoadMetrics, CompletionCallback&& completionCallback)
    {
        return adoptRef(*new NetworkCORSPreflightChecker(networkProcess, networkResourceLoader, WTFMove(parameters), shouldCaptureExtraNetworkLoadMetrics, WTFMove(completionCallback)));
    }
    ~NetworkCORSPreflightChecker();
    const WebCore::ResourceRequest& originalRequest() const { return m_parameters.originalRequest; }

    void startPreflight();

    WebCore::NetworkTransactionInformation takeInformation();

private:
    NetworkCORSPreflightChecker(NetworkProcess&, NetworkResourceLoader*, Parameters&&, bool shouldCaptureExtraNetworkLoadMetrics, CompletionCallback&&);

    void willPerformHTTPRedirection(WebCore::ResourceResponse&&, WebCore::ResourceRequest&&, RedirectCompletionHandler&&) final;
    void didReceiveChallenge(WebCore::AuthenticationChallenge&&, NegotiatedLegacyTLS, ChallengeCompletionHandler&&) final;
    void didReceiveResponse(WebCore::ResourceResponse&&, NegotiatedLegacyTLS, PrivateRelayed, ResponseCompletionHandler&&) final;
    void didReceiveData(const WebCore::SharedBuffer&) final;
    void didCompleteWithError(const WebCore::ResourceError&, const WebCore::NetworkLoadMetrics&) final;
    void didSendData(uint64_t totalBytesSent, uint64_t totalBytesExpectedToSend) final;
    void wasBlocked() final;
    void cannotShowURL() final;
    void wasBlockedByRestrictions() final;
    void wasBlockedByDisabledFTP() final;

    Parameters m_parameters;
    Ref<NetworkProcess> m_networkProcess;
    WebCore::ResourceResponse m_response;
    CompletionCallback m_completionCallback;
    RefPtr<NetworkDataTask> m_task;
    bool m_shouldCaptureExtraNetworkLoadMetrics { false };
    WebCore::NetworkTransactionInformation m_loadInformation;
    WeakPtr<NetworkResourceLoader> m_networkResourceLoader;
};

} // namespace WebKit
