/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#include "config.h"
#include "PendingDownload.h"

#include "Download.h"
#include "DownloadProxyMessages.h"
#include "MessageSenderInlines.h"
#include "NetworkConnectionToWebProcess.h"
#include "NetworkLoad.h"
#include "NetworkProcess.h"
#include "NetworkSession.h"
#include <WebCore/LocalFrameLoaderClient.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PendingDownload);

PendingDownload::PendingDownload(IPC::Connection* parentProcessConnection, NetworkLoadParameters&& parameters, DownloadID downloadID, NetworkSession& networkSession, const String& suggestedName, FromDownloadAttribute fromDownloadAttribute, std::optional<WebCore::ProcessIdentifier> webProcessID)
    : m_networkLoad(NetworkLoad::create(*this, WTFMove(parameters), networkSession))
    , m_parentProcessConnection(parentProcessConnection)
    , m_fromDownloadAttribute(fromDownloadAttribute)
    , m_webProcessID(webProcessID)
{
    m_networkLoad->start();
    m_isAllowedToAskUserForCredentials = parameters.clientCredentialPolicy == ClientCredentialPolicy::MayAskClientForCredentials;

    m_networkLoad->setPendingDownloadID(downloadID);
    m_networkLoad->setPendingDownload(*this);
    m_networkLoad->setSuggestedFilename(suggestedName);

    send(Messages::DownloadProxy::DidStart(m_networkLoad->currentRequest(), suggestedName));
}

PendingDownload::PendingDownload(IPC::Connection* parentProcessConnection, Ref<NetworkLoad>&& networkLoad, ResponseCompletionHandler&& completionHandler, DownloadID downloadID, const ResourceRequest& request, const ResourceResponse& response)
    : m_networkLoad(WTFMove(networkLoad))
    , m_parentProcessConnection(parentProcessConnection)
{
    m_isAllowedToAskUserForCredentials = m_networkLoad->isAllowedToAskUserForCredentials();

    m_networkLoad->setPendingDownloadID(downloadID);
    send(Messages::DownloadProxy::DidStart(request, String()));

    m_networkLoad->convertTaskToDownload(*this, request, response, WTFMove(completionHandler));
}

PendingDownload::~PendingDownload() = default;

bool PendingDownload::isDownloadTriggeredWithDownloadAttribute() const
{
    return m_fromDownloadAttribute == FromDownloadAttribute::Yes;
}

inline static bool isRedirectCrossOrigin(const WebCore::ResourceRequest& redirectRequest, const WebCore::ResourceResponse& redirectResponse)
{
    return !SecurityOrigin::create(redirectResponse.url())->isSameOriginAs(SecurityOrigin::create(redirectRequest.url()));
}

void PendingDownload::willSendRedirectedRequest(WebCore::ResourceRequest&&, WebCore::ResourceRequest&& redirectRequest, WebCore::ResourceResponse&& redirectResponse, CompletionHandler<void(WebCore::ResourceRequest&&)>&& completionHandler)
{
#if PLATFORM(COCOA)
    bool linkedOnOrAfterBlockCrossOriginDownloads = WTF::linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::BlockCrossOriginRedirectDownloads);
#else
    bool linkedOnOrAfterBlockCrossOriginDownloads = true;
#endif
    if (linkedOnOrAfterBlockCrossOriginDownloads && isDownloadTriggeredWithDownloadAttribute() && isRedirectCrossOrigin(redirectRequest, redirectResponse)) {
        completionHandler(WebCore::ResourceRequest());
        m_networkLoad->cancel();
        if (m_webProcessID && !redirectRequest.url().protocolIsJavaScript() && m_networkLoad->webFrameID() && m_networkLoad->webPageID())
            m_networkLoad->networkProcess()->webProcessConnection(*m_webProcessID)->loadCancelledDownloadRedirectRequestInFrame(redirectRequest, *m_networkLoad->webFrameID(), *m_networkLoad->webPageID());
        return;
    }
    sendWithAsyncReply(Messages::DownloadProxy::WillSendRequest(WTFMove(redirectRequest), WTFMove(redirectResponse)), WTFMove(completionHandler));
};

void PendingDownload::cancel(CompletionHandler<void(std::span<const uint8_t>)>&& completionHandler)
{
    m_networkLoad->cancel();
    completionHandler({ });
}

#if PLATFORM(COCOA)
#if HAVE(MODERN_DOWNLOADPROGRESS)
void PendingDownload::publishProgress(const URL& url, std::span<const uint8_t> bookmarkData, UseDownloadPlaceholder useDownloadPlaceholder, std::span<const uint8_t> activityAccessToken)
{
    ASSERT(!m_progressURL.isValid());
    m_progressURL = url;
    m_bookmarkData = bookmarkData;
    m_useDownloadPlaceholder = useDownloadPlaceholder;
    m_activityAccessToken = activityAccessToken;
}
#else
void PendingDownload::publishProgress(const URL& url, SandboxExtension::Handle&& sandboxExtension)
{
    ASSERT(!m_progressURL.isValid());
    m_progressURL = url;
    m_progressSandboxExtension = WTFMove(sandboxExtension);
}
#endif

void PendingDownload::didBecomeDownload(Download& download)
{
    if (!m_progressURL.isValid())
        return;
#if HAVE(MODERN_DOWNLOADPROGRESS)
    download.publishProgress(m_progressURL, m_bookmarkData, m_useDownloadPlaceholder, m_activityAccessToken);
#else
    download.publishProgress(m_progressURL, WTFMove(m_progressSandboxExtension));
#endif
}
#endif // PLATFORM(COCOA)

void PendingDownload::didFailLoading(const WebCore::ResourceError& error)
{
    // FIXME: For Cross Origin redirects Cancellation happens early. So avoid repeating. Maybe there is a better way ?
    if (!m_isDownloadCancelled) {
        m_isDownloadCancelled = true;
        send(Messages::DownloadProxy::DidFail(error, { }));
    }
}
    
IPC::Connection* PendingDownload::messageSenderConnection() const
{
    return m_parentProcessConnection.get();
}

void PendingDownload::didReceiveResponse(WebCore::ResourceResponse&& response, PrivateRelayed, ResponseCompletionHandler&& completionHandler)
{
    completionHandler(WebCore::PolicyAction::Download);
}

uint64_t PendingDownload::messageSenderDestinationID() const
{
    return m_networkLoad->pendingDownloadID()->toUInt64();
}
    
}
