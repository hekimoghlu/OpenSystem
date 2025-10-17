/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
#include <WebCore/BackgroundFetchRecordLoader.h>
#include <WebCore/ResourceError.h>
#include <WebCore/ResourceRequest.h>
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class ResourceRequest;
struct BackgroundFetchRequest;
struct ClientOrigin;
struct FetchOptions;
}

namespace WebKit {

class NetworkLoadChecker;
class NetworkProcess;

class BackgroundFetchLoad final : public RefCounted<BackgroundFetchLoad>, public WebCore::BackgroundFetchRecordLoader, public NetworkDataTaskClient {
    WTF_MAKE_TZONE_ALLOCATED(BackgroundFetchLoad);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<BackgroundFetchLoad> create(NetworkProcess& networkProcess, PAL::SessionID sessionID,
        WebCore::BackgroundFetchRecordLoaderClient& backgroundFetchRecordLoaderClient,
        const WebCore::BackgroundFetchRequest& backgroundFetchRequest, size_t responseDataSize,
        const WebCore::ClientOrigin& clientOrigin)
    {
        return adoptRef(*new BackgroundFetchLoad(networkProcess, sessionID, backgroundFetchRecordLoaderClient, backgroundFetchRequest, responseDataSize, clientOrigin));
    }

    ~BackgroundFetchLoad();

private:
    BackgroundFetchLoad(NetworkProcess&, PAL::SessionID, WebCore::BackgroundFetchRecordLoaderClient&, const WebCore::BackgroundFetchRequest&, size_t responseDataSize, const WebCore::ClientOrigin&);

    const URL& currentURL() const;

    // NetworkDataTaskClient
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

    // WebCore::BackgroundFetchRecordLoader
    void abort() final;

    void loadRequest(NetworkProcess&, WebCore::ResourceRequest&&);

    void didFinish(const WebCore::ResourceError& = { }, const WebCore::ResourceResponse& response = { });

    PAL::SessionID m_sessionID;
    WeakPtr<WebCore::BackgroundFetchRecordLoaderClient> m_client;
    WebCore::ResourceRequest m_request;
    RefPtr<NetworkDataTask> m_task;
    Ref<NetworkLoadChecker> m_networkLoadChecker;
    Vector<RefPtr<WebCore::BlobDataFileReference>> m_blobFiles;
};

}
