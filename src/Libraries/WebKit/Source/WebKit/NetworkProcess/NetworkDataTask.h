/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
#include "SandboxExtension.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/Credential.h>
#include <WebCore/FrameLoaderTypes.h>
#include <WebCore/NetworkLoadMetrics.h>
#include <WebCore/ResourceLoaderOptions.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/StoredCredentialsPolicy.h>
#include <pal/SessionID.h>
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/CompletionHandler.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class AuthenticationChallenge;
class IPAddress;
class ResourceError;
class ResourceResponse;
class SharedBuffer;
}

namespace WebKit {

class NetworkSession;
class PendingDownload;

enum class AuthenticationChallengeDisposition : uint8_t;
enum class NegotiatedLegacyTLS : bool;
enum class PrivateRelayed : bool;

struct NetworkLoadParameters;

using RedirectCompletionHandler = CompletionHandler<void(WebCore::ResourceRequest&&)>;
using ChallengeCompletionHandler = CompletionHandler<void(AuthenticationChallengeDisposition, const WebCore::Credential&)>;
using ResponseCompletionHandler = CompletionHandler<void(WebCore::PolicyAction)>;

class NetworkDataTaskClient : public AbstractRefCountedAndCanMakeWeakPtr<NetworkDataTaskClient> {
public:
    virtual void willPerformHTTPRedirection(WebCore::ResourceResponse&&, WebCore::ResourceRequest&&, RedirectCompletionHandler&&) = 0;
    virtual void didReceiveChallenge(WebCore::AuthenticationChallenge&&, NegotiatedLegacyTLS, ChallengeCompletionHandler&&) = 0;
    virtual void didReceiveInformationalResponse(WebCore::ResourceResponse&&) { };
    virtual void didReceiveResponse(WebCore::ResourceResponse&&, NegotiatedLegacyTLS, PrivateRelayed, ResponseCompletionHandler&&) = 0;
    virtual void didReceiveData(const WebCore::SharedBuffer&) = 0;
    virtual void didCompleteWithError(const WebCore::ResourceError&, const WebCore::NetworkLoadMetrics&) = 0;
    virtual void didSendData(uint64_t totalBytesSent, uint64_t totalBytesExpectedToSend) = 0;
    virtual void wasBlocked() = 0;
    virtual void cannotShowURL() = 0;
    virtual void wasBlockedByRestrictions() = 0;
    virtual void wasBlockedByDisabledFTP() = 0;

    virtual bool shouldCaptureExtraNetworkLoadMetrics() const { return false; }

    virtual void didNegotiateModernTLS(const URL&) { }

    void didCompleteWithError(const WebCore::ResourceError& error)
    {
        WebCore::NetworkLoadMetrics emptyMetrics;
        didCompleteWithError(error, emptyMetrics);
    }

    virtual ~NetworkDataTaskClient() { }
};

class NetworkDataTask : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<NetworkDataTask, WTF::DestructionThread::Main> {
public:
    static Ref<NetworkDataTask> create(NetworkSession&, NetworkDataTaskClient&, const NetworkLoadParameters&);

    virtual ~NetworkDataTask();

    virtual void cancel() = 0;
    virtual void resume() = 0;
    virtual void invalidateAndCancel() = 0;

    void didReceiveInformationalResponse(WebCore::ResourceResponse&&);
    void didReceiveResponse(WebCore::ResourceResponse&&, NegotiatedLegacyTLS, PrivateRelayed, std::optional<WebCore::IPAddress>, ResponseCompletionHandler&&);
    bool shouldCaptureExtraNetworkLoadMetrics() const;

    enum class State {
        Running,
        Suspended,
        Canceling,
        Completed
    };
    virtual State state() const = 0;

    NetworkDataTaskClient* client() const { return m_client.get(); }
    void clearClient() { m_client = nullptr; }

    std::optional<DownloadID> pendingDownloadID() const { return m_pendingDownloadID.asOptional(); }
    PendingDownload* pendingDownload() const;
    void setPendingDownloadID(DownloadID downloadID)
    {
        ASSERT(!m_pendingDownloadID);
        m_pendingDownloadID = downloadID;
    }
    void setPendingDownload(PendingDownload&);

    virtual void setPendingDownloadLocation(const String& filename, SandboxExtension::Handle&&, bool /*allowOverwrite*/) { m_pendingDownloadLocation = filename; }
    const String& pendingDownloadLocation() const { return m_pendingDownloadLocation; }
    bool isDownload() const { return !!m_pendingDownloadID; }

    const WebCore::ResourceRequest& firstRequest() const { return m_firstRequest; }
    virtual String suggestedFilename() const { return String(); }
    void setSuggestedFilename(const String& suggestedName) { m_suggestedFilename = suggestedName; }
    const String& partition() { return m_partition; }

    bool isTopLevelNavigation() const { return m_dataTaskIsForMainFrameNavigation; }

    virtual String description() const;
    virtual void setH2PingCallback(const URL&, CompletionHandler<void(Expected<WTF::Seconds, WebCore::ResourceError>&&)>&&);

    virtual void setPriority(WebCore::ResourceLoadPriority) { }
    String attributedBundleIdentifier(WebPageProxyIdentifier);

#if ENABLE(INSPECTOR_NETWORK_THROTTLING)
    virtual void setEmulatedConditions(const std::optional<int64_t>& /* bytesPerSecondLimit */) { }
#endif

    PAL::SessionID sessionID() const;

    const NetworkSession* networkSession() const;
    NetworkSession* networkSession();

    virtual void setTimingAllowFailedFlag() { }

protected:
    NetworkDataTask(NetworkSession&, NetworkDataTaskClient&, const WebCore::ResourceRequest&, WebCore::StoredCredentialsPolicy, bool shouldClearReferrerOnHTTPSToHTTPRedirect, bool dataTaskIsForMainFrameNavigation);

    enum class FailureType : uint8_t {
        Blocked,
        InvalidURL,
        RestrictedURL,
        FTPDisabled
    };
    void scheduleFailure(FailureType);

    void restrictRequestReferrerToOriginIfNeeded(WebCore::ResourceRequest&);

    WeakPtr<NetworkSession> m_session;
    WeakPtr<NetworkDataTaskClient> m_client;
    WeakPtr<PendingDownload> m_pendingDownload;
    Markable<DownloadID> m_pendingDownloadID;
    String m_user;
    String m_password;
    String m_partition;
    WebCore::Credential m_initialCredential;
    WebCore::StoredCredentialsPolicy m_storedCredentialsPolicy { WebCore::StoredCredentialsPolicy::DoNotUse };
    String m_lastHTTPMethod;
    String m_pendingDownloadLocation;
    WebCore::ResourceRequest m_firstRequest;
    WebCore::ResourceRequest m_previousRequest;
    bool m_shouldClearReferrerOnHTTPSToHTTPRedirect { true };
    String m_suggestedFilename;
    bool m_dataTaskIsForMainFrameNavigation { false };
    bool m_failureScheduled { false };
};

} // namespace WebKit
