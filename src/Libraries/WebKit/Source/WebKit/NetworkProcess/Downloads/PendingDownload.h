/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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
#include "MessageSender.h"
#include "NetworkLoadClient.h"
#include "SandboxExtension.h"
#include "UseDownloadPlaceholder.h"
#include <WebCore/ProcessIdentifier.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>

namespace IPC {
class Connection;
}

namespace WebCore {
class ResourceResponse;

enum class FromDownloadAttribute : bool;
}

namespace WebKit {

class Download;
class NetworkLoad;
class NetworkSession;

struct NetworkLoadParameters;

class PendingDownload : public RefCountedAndCanMakeWeakPtr<PendingDownload>, public NetworkLoadClient, public IPC::MessageSender {
    WTF_MAKE_TZONE_ALLOCATED(PendingDownload);
public:
    static Ref<PendingDownload> create(IPC::Connection* connection, NetworkLoadParameters&& networkLoadParameters, DownloadID downloadID, NetworkSession& networkSession, const String& suggestedName, WebCore::FromDownloadAttribute fromDownloadAttribute, std::optional<WebCore::ProcessIdentifier> webProcessId)
    {
        return adoptRef(*new PendingDownload(connection, WTFMove(networkLoadParameters), downloadID, networkSession, suggestedName, fromDownloadAttribute, webProcessId));
    }

    static Ref<PendingDownload> create(IPC::Connection* connection, Ref<NetworkLoad>&& networkLoad, ResponseCompletionHandler&& responseCompletionHandler, DownloadID downloadID, const WebCore::ResourceRequest& resourceRequest, const WebCore::ResourceResponse& resourceResponse)
    {
        return adoptRef(*new PendingDownload(connection, WTFMove(networkLoad), WTFMove(responseCompletionHandler), downloadID, resourceRequest, resourceResponse));
    }

    virtual ~PendingDownload();

    void cancel(CompletionHandler<void(std::span<const uint8_t>)>&&);

#if PLATFORM(COCOA)
#if HAVE(MODERN_DOWNLOADPROGRESS)
    void publishProgress(const URL&, std::span<const uint8_t>, UseDownloadPlaceholder, std::span<const uint8_t>);
#else
    void publishProgress(const URL&, SandboxExtension::Handle&&);
#endif
    void didBecomeDownload(Download&);
#endif

private:    
    PendingDownload(IPC::Connection*, NetworkLoadParameters&&, DownloadID, NetworkSession&, const String& suggestedName, WebCore::FromDownloadAttribute, std::optional<WebCore::ProcessIdentifier>);
    PendingDownload(IPC::Connection*, Ref<NetworkLoad>&&, ResponseCompletionHandler&&, DownloadID, const WebCore::ResourceRequest&, const WebCore::ResourceResponse&);

    // NetworkLoadClient.
    void didSendData(uint64_t bytesSent, uint64_t totalBytesToBeSent) override { }
    bool isSynchronous() const override { return false; }
    bool isAllowedToAskUserForCredentials() const final { return m_isAllowedToAskUserForCredentials; }
    void willSendRedirectedRequest(WebCore::ResourceRequest&&, WebCore::ResourceRequest&& redirectRequest, WebCore::ResourceResponse&& redirectResponse, CompletionHandler<void(WebCore::ResourceRequest&&)>&&) override;
    void didReceiveResponse(WebCore::ResourceResponse&&, PrivateRelayed, ResponseCompletionHandler&&) override;
    void didReceiveBuffer(const WebCore::FragmentedSharedBuffer&, uint64_t reportedEncodedDataLength) override { };
    void didFinishLoading(const WebCore::NetworkLoadMetrics&) override { };
    void didFailLoading(const WebCore::ResourceError&) override;
    bool isDownloadTriggeredWithDownloadAttribute() const;

    // MessageSender.
    IPC::Connection* messageSenderConnection() const override;
    uint64_t messageSenderDestinationID() const override;

private:
    Ref<NetworkLoad> m_networkLoad;
    RefPtr<IPC::Connection> m_parentProcessConnection;
    bool m_isAllowedToAskUserForCredentials;
    bool m_isDownloadCancelled = false;
    WebCore::FromDownloadAttribute m_fromDownloadAttribute;
    std::optional<WebCore::ProcessIdentifier> m_webProcessID;

#if PLATFORM(COCOA)
    URL m_progressURL;
#if HAVE(MODERN_DOWNLOADPROGRESS)
    Vector<uint8_t> m_bookmarkData;
    Vector<uint8_t> m_activityAccessToken;
    UseDownloadPlaceholder m_useDownloadPlaceholder { UseDownloadPlaceholder::No };
#else
    SandboxExtension::Handle m_progressSandboxExtension;
#endif
#endif
};

}
