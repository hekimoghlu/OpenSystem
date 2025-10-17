/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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

#include "DownloadID.h"
#include "NetworkDataTask.h"
#include "NetworkLoadParameters.h"
#include "NetworkSession.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class AuthenticationChallenge;
}

namespace WebKit {

class NetworkLoadClient;
class NetworkLoadScheduler;
class NetworkProcess;

class NetworkLoad final : public RefCounted<NetworkLoad>, public NetworkDataTaskClient {
    WTF_MAKE_TZONE_ALLOCATED(NetworkLoad);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<NetworkLoad> create(NetworkLoadClient& networkLoadClient, NetworkLoadParameters&& networkLoadParameters, NetworkSession& networkSession)
    {
        return adoptRef(*new NetworkLoad(networkLoadClient, WTFMove(networkLoadParameters), networkSession));
    }

    template<typename CreateTaskCallback> static Ref<NetworkLoad> create(NetworkLoadClient& networkLoadClient, NetworkSession& networkSession, NOESCAPE const CreateTaskCallback& createTask)
    {
        return adoptRef(*new NetworkLoad(networkLoadClient, networkSession, createTask));
    }

    ~NetworkLoad();

    void start();
    void startWithScheduling();
    void cancel();

    bool isAllowedToAskUserForCredentials() const;

    const WebCore::ResourceRequest& currentRequest() const { return m_currentRequest; }
    void updateRequestAfterRedirection(WebCore::ResourceRequest&) const;
    void reprioritizeRequest(WebCore::ResourceLoadPriority);

    const NetworkLoadParameters& parameters() const { return m_parameters; }
    const URL& url() const { return parameters().request.url(); }
    String attributedBundleIdentifier(WebPageProxyIdentifier);

    void convertTaskToDownload(PendingDownload&, const WebCore::ResourceRequest&, const WebCore::ResourceResponse&, ResponseCompletionHandler&&);
    void setPendingDownloadID(DownloadID);
    void setSuggestedFilename(const String&);
    void setPendingDownload(PendingDownload&);
    std::optional<DownloadID> pendingDownloadID() { return protectedTask()->pendingDownloadID(); }

    bool shouldCaptureExtraNetworkLoadMetrics() const final;

    String description() const;
    void setH2PingCallback(const URL&, CompletionHandler<void(Expected<WTF::Seconds, WebCore::ResourceError>&&)>&&);

    void setTimingAllowFailedFlag();
    std::optional<WebCore::FrameIdentifier> webFrameID() const;
    std::optional<WebCore::PageIdentifier> webPageID() const;
    Ref<NetworkProcess> networkProcess();

private:
    NetworkLoad(NetworkLoadClient&, NetworkLoadParameters&&, NetworkSession&);

    template<typename CreateTaskCallback> NetworkLoad(NetworkLoadClient& client, NetworkSession& networkSession, NOESCAPE const CreateTaskCallback& createTask)
        : m_client(client)
        , m_networkProcess(networkSession.networkProcess())
        , m_task(createTask(*this))
    {
    }

    // NetworkDataTaskClient
    void willPerformHTTPRedirection(WebCore::ResourceResponse&&, WebCore::ResourceRequest&&, RedirectCompletionHandler&&) final;
    void didReceiveChallenge(WebCore::AuthenticationChallenge&&, NegotiatedLegacyTLS, ChallengeCompletionHandler&&) final;
    void didReceiveInformationalResponse(WebCore::ResourceResponse&&) final;
    void didReceiveResponse(WebCore::ResourceResponse&&, NegotiatedLegacyTLS, PrivateRelayed, ResponseCompletionHandler&&) final;
    void didReceiveData(const WebCore::SharedBuffer&) final;
    void didCompleteWithError(const WebCore::ResourceError&, const WebCore::NetworkLoadMetrics&) final;
    void didSendData(uint64_t totalBytesSent, uint64_t totalBytesExpectedToSend) final;
    void wasBlocked() final;
    void cannotShowURL() final;
    void wasBlockedByRestrictions() final;
    void wasBlockedByDisabledFTP() final;
    void didNegotiateModernTLS(const URL&) final;

    RefPtr<NetworkDataTask> protectedTask();

    void notifyDidReceiveResponse(WebCore::ResourceResponse&&, NegotiatedLegacyTLS, PrivateRelayed, ResponseCompletionHandler&&);

    std::reference_wrapper<NetworkLoadClient> m_client;
    Ref<NetworkProcess> m_networkProcess;
    const NetworkLoadParameters m_parameters;
    RefPtr<NetworkDataTask> m_task;
    WeakPtr<NetworkLoadScheduler> m_scheduler;

    // FIXME: Deduplicate this with NetworkDataTask's m_previousRequest.
    WebCore::ResourceRequest m_currentRequest; // Updated on redirects.
};

} // namespace WebKit
