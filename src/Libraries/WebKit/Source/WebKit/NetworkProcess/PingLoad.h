/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
#include "NetworkResourceLoadParameters.h"
#include <WebCore/ResourceError.h>
#include <WebCore/ResourceResponse.h>
#include <wtf/CompletionHandler.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class NetworkConnectionToWebProcess;
class NetworkLoadChecker;
class NetworkProcess;
class NetworkSchemeRegistry;

class PingLoad final : public RefCounted<PingLoad>, public NetworkDataTaskClient {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static void create(NetworkProcess& networkProcess, PAL::SessionID sessionID, NetworkResourceLoadParameters&& networkResourceLoadParameters, CompletionHandler<void(const WebCore::ResourceError&, const WebCore::ResourceResponse&)>&& completionHandler)
    {
        auto pingLoad = new PingLoad(networkProcess, sessionID, WTFMove(networkResourceLoadParameters), WTFMove(completionHandler));

        // Keep the load alive until didFinish.
        pingLoad->m_selfReference = adoptRef(pingLoad);
    }

    static void create(NetworkConnectionToWebProcess& networkConnectionToWebProcess, NetworkResourceLoadParameters&& networkResourceLoadParameters, CompletionHandler<void(const WebCore::ResourceError&, const WebCore::ResourceResponse&)>&& completionHandler)
    {
        auto pingLoad = adoptRef(*new PingLoad(networkConnectionToWebProcess, WTFMove(networkResourceLoadParameters), WTFMove(completionHandler)));

        // Keep the load alive until didFinish.
        pingLoad->m_selfReference = WTFMove(pingLoad);
    }

    ~PingLoad();

private:
    PingLoad(NetworkProcess&, PAL::SessionID, NetworkResourceLoadParameters&&, CompletionHandler<void(const WebCore::ResourceError&, const WebCore::ResourceResponse&)>&&);
    PingLoad(NetworkConnectionToWebProcess&, NetworkResourceLoadParameters&&, CompletionHandler<void(const WebCore::ResourceError&, const WebCore::ResourceResponse&)>&&);

    void initialize(NetworkProcess&);

    const URL& currentURL() const;

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
    void timeoutTimerFired();

    void loadRequest(NetworkProcess&, WebCore::ResourceRequest&&);

    void didFinish(const WebCore::ResourceError& = { }, const WebCore::ResourceResponse& response = { });
    
    RefPtr<PingLoad> m_selfReference;
    PAL::SessionID m_sessionID;
    NetworkResourceLoadParameters m_parameters;
    CompletionHandler<void(const WebCore::ResourceError&, const WebCore::ResourceResponse&)> m_completionHandler;
    RefPtr<NetworkDataTask> m_task;
    WebCore::Timer m_timeoutTimer;
    Ref<NetworkLoadChecker> m_networkLoadChecker;
    Vector<RefPtr<WebCore::BlobDataFileReference>> m_blobFiles;
};

}
