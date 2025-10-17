/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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

#include "NetworkActivityTracker.h"
#include "NetworkDataTask.h"
#include "NetworkLoadParameters.h"
#include "NetworkTaskCocoa.h"
#include <WebCore/NetworkLoadMetrics.h>
#include <WebCore/PrivateClickMeasurement.h>
#include <wtf/RetainPtr.h>

OBJC_CLASS NSHTTPCookieStorage;
OBJC_CLASS NSURLSessionDataTask;
OBJC_CLASS NSMutableURLRequest;

namespace WebCore {
class RegistrableDomain;
class SharedBuffer;
enum class AdvancedPrivacyProtections : uint16_t;
}

namespace WebKit {

class Download;
class NetworkSessionCocoa;
struct SessionWrapper;

class NetworkDataTaskCocoa final : public NetworkDataTask, public NetworkTaskCocoa {
public:
    static Ref<NetworkDataTask> create(NetworkSession& session, NetworkDataTaskClient& client, const NetworkLoadParameters& parameters)
    {
        return adoptRef(*new NetworkDataTaskCocoa(session, client, parameters));
    }

    ~NetworkDataTaskCocoa();

    using TaskIdentifier = uint64_t;

    void didSendData(uint64_t totalBytesSent, uint64_t totalBytesExpectedToSend);
    void didReceiveChallenge(WebCore::AuthenticationChallenge&&, NegotiatedLegacyTLS, ChallengeCompletionHandler&&);
    void didNegotiateModernTLS(const URL&);
    void didCompleteWithError(const WebCore::ResourceError&, const WebCore::NetworkLoadMetrics&);
    void didReceiveResponse(WebCore::ResourceResponse&&, NegotiatedLegacyTLS, PrivateRelayed, ResponseCompletionHandler&&);
    void didReceiveData(const WebCore::SharedBuffer&);

    void willPerformHTTPRedirection(WebCore::ResourceResponse&&, WebCore::ResourceRequest&&, RedirectCompletionHandler&&);
    void transferSandboxExtensionToDownload(Download&);

    void cancel() override;
    void resume() override;
    void invalidateAndCancel() override { }
    NetworkDataTask::State state() const override;

    void setPendingDownloadLocation(const String&, SandboxExtension::Handle&&, bool /*allowOverwrite*/) override;
    String suggestedFilename() const override;

    WebCore::NetworkLoadMetrics& networkLoadMetrics() { return m_networkLoadMetrics; }

    std::optional<WebCore::FrameIdentifier> frameID() const final { return m_frameID; }
    std::optional<WebCore::PageIdentifier> pageID() const final { return m_pageID; }
    std::optional<WebPageProxyIdentifier> webPageProxyID() const final { return m_webPageProxyID; }

    String description() const override;

    void setH2PingCallback(const URL&, CompletionHandler<void(Expected<WTF::Seconds, WebCore::ResourceError>&&)>&&) override;
    void setPriority(WebCore::ResourceLoadPriority) override;
#if ENABLE(INSPECTOR_NETWORK_THROTTLING)
    void setEmulatedConditions(const std::optional<int64_t>& bytesPerSecondLimit) override;
#endif

    void checkTAO(const WebCore::ResourceResponse&);

private:
    NetworkDataTaskCocoa(NetworkSession&, NetworkDataTaskClient&, const NetworkLoadParameters&);

    bool tryPasswordBasedAuthentication(const WebCore::AuthenticationChallenge&, ChallengeCompletionHandler&);
    void applySniffingPoliciesAndBindRequestToInferfaceIfNeeded(RetainPtr<NSURLRequest>&, bool shouldContentSniff, WebCore::ContentEncodingSniffingPolicy);

    void updateFirstPartyInfoForSession(const URL&);

    NSURLSessionTask* task() const final;
    WebCore::StoredCredentialsPolicy storedCredentialsPolicy() const final { return m_storedCredentialsPolicy; }

    void setTimingAllowFailedFlag() final;

    WeakPtr<SessionWrapper> m_sessionWrapper;
    RefPtr<SandboxExtension> m_sandboxExtension;
    RetainPtr<NSURLSessionDataTask> m_task;
    WebCore::NetworkLoadMetrics m_networkLoadMetrics;
    Markable<WebCore::FrameIdentifier> m_frameID;
    Markable<WebCore::PageIdentifier> m_pageID;
    Markable<WebPageProxyIdentifier> m_webPageProxyID;

    bool m_isForMainResourceNavigationForAnyFrame { false };
    RefPtr<WebCore::SecurityOrigin> m_sourceOrigin;
};

WebCore::Credential serverTrustCredential(const WebCore::AuthenticationChallenge&);
void setPCMDataCarriedOnRequest(WebCore::PrivateClickMeasurement::PcmDataCarried, NSMutableURLRequest *);

void enableAdvancedPrivacyProtections(NSMutableURLRequest *, OptionSet<WebCore::AdvancedPrivacyProtections>);

} // namespace WebKit
