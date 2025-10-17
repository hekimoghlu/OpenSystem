/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
#include "DownloadManager.h"

#include "Download.h"
#include "DownloadProxyMessages.h"
#include "MessageSenderInlines.h"
#include "NetworkConnectionToWebProcess.h"
#include "NetworkLoad.h"
#include "NetworkSession.h"
#include "PendingDownload.h"
#include <WebCore/NotImplemented.h>
#include <pal/SessionID.h>
#include <wtf/StdLibExtras.h>

namespace WebKit {
using namespace WebCore;

DownloadManager::DownloadManager(Client& client)
    : m_client(client)
{
}

DownloadManager::~DownloadManager() = default;

void DownloadManager::startDownload(PAL::SessionID sessionID, DownloadID downloadID, const ResourceRequest& request, const std::optional<WebCore::SecurityOriginData>& topOrigin, std::optional<NavigatingToAppBoundDomain> isNavigatingToAppBoundDomain, const String& suggestedName, FromDownloadAttribute fromDownloadAttribute, std::optional<WebCore::FrameIdentifier> frameID, std::optional<WebCore::PageIdentifier> pageID, std::optional<WebCore::ProcessIdentifier> webProcessID)
{
    auto* networkSession = client().networkSession(sessionID);
    if (!networkSession)
        return;

    NetworkLoadParameters parameters;
    if (frameID)
        parameters.webFrameID = *frameID;
    if (pageID)
        parameters.webPageID = *pageID;
    parameters.request = request;
    parameters.clientCredentialPolicy = ClientCredentialPolicy::MayAskClientForCredentials;
    parameters.isNavigatingToAppBoundDomain = isNavigatingToAppBoundDomain;
    if (request.url().protocolIsBlob()) {
        parameters.topOrigin = topOrigin ? topOrigin->securityOrigin().ptr() : nullptr;
        parameters.blobFileReferences = client().networkSession(sessionID)->blobRegistry().filesInBlob(request.url(), topOrigin);
    }
    parameters.storedCredentialsPolicy = sessionID.isEphemeral() ? StoredCredentialsPolicy::DoNotUse : StoredCredentialsPolicy::Use;

    m_pendingDownloads.add(downloadID, PendingDownload::create(m_client->parentProcessConnectionForDownloads(), WTFMove(parameters), downloadID, *networkSession, suggestedName, fromDownloadAttribute, webProcessID));
}

void DownloadManager::dataTaskBecameDownloadTask(DownloadID downloadID, Ref<Download>&& download)
{
    ASSERT(m_pendingDownloads.contains(downloadID));
    if (RefPtr pendingDownload = m_pendingDownloads.take(downloadID)) {
#if PLATFORM(COCOA)
        pendingDownload->didBecomeDownload(download);
#endif
    }
    ASSERT(!m_downloads.contains(downloadID));
    m_downloadsAfterDestinationDecided.remove(downloadID);
    m_downloads.add(downloadID, WTFMove(download));
}

void DownloadManager::convertNetworkLoadToDownload(DownloadID downloadID, Ref<NetworkLoad>&& networkLoad, ResponseCompletionHandler&& completionHandler, Vector<RefPtr<WebCore::BlobDataFileReference>>&& blobFileReferences, const ResourceRequest& request, const ResourceResponse& response)
{
    ASSERT(!m_pendingDownloads.contains(downloadID));
    m_pendingDownloads.add(downloadID, PendingDownload::create(m_client->parentProcessConnectionForDownloads(), WTFMove(networkLoad), WTFMove(completionHandler), downloadID, request, response));
}

void DownloadManager::downloadDestinationDecided(DownloadID downloadID, Ref<NetworkDataTask>&& networkDataTask)
{
    ASSERT(!m_downloadsAfterDestinationDecided.contains(downloadID));
    m_downloadsAfterDestinationDecided.set(downloadID, WTFMove(networkDataTask));
}

void DownloadManager::resumeDownload(PAL::SessionID sessionID, DownloadID downloadID, std::span<const uint8_t> resumeData, const String& path, SandboxExtension::Handle&& sandboxExtensionHandle, CallDownloadDidStart callDownloadDidStart, std::span<const uint8_t> activityAccessToken)
{
#if !PLATFORM(COCOA)
    notImplemented();
#else
    auto* networkSession = m_client->networkSession(sessionID);
    if (!networkSession)
        return;
    Ref download = Download::create(*this, downloadID, nullptr, *networkSession);

    download->resume(resumeData, path, WTFMove(sandboxExtensionHandle), activityAccessToken);

    // For compatibility with the legacy download API, only send DidStart if we're using the new API.
    if (callDownloadDidStart == CallDownloadDidStart::Yes)
        download->send(Messages::DownloadProxy::DidStart({ }, { }));

    ASSERT(!m_downloads.contains(downloadID));
    m_downloads.add(downloadID, WTFMove(download));
#endif
}

void DownloadManager::cancelDownload(DownloadID downloadID, CompletionHandler<void(std::span<const uint8_t>)>&& completionHandler)
{
    if (RefPtr download = m_downloads.get(downloadID)) {
        ASSERT(!m_pendingDownloads.contains(downloadID));
        download->cancel(WTFMove(completionHandler), Download::IgnoreDidFailCallback::Yes);
        return;
    }
    if (RefPtr pendingDownload = m_pendingDownloads.take(downloadID)) {
        pendingDownload->cancel(WTFMove(completionHandler));
        return;
    }
    // If there is no active or pending download, then the download finished in a short race window after cancellation was requested.
    completionHandler({ });
}

Download* DownloadManager::download(DownloadID downloadID)
{
    return m_downloads.get(downloadID);
}

#if PLATFORM(COCOA)
#if HAVE(MODERN_DOWNLOADPROGRESS)
void DownloadManager::publishDownloadProgress(DownloadID downloadID, const URL& url, std::span<const uint8_t> bookmarkData, WebKit::UseDownloadPlaceholder useDownloadPlaceholder, std::span<const uint8_t> activityAccessToken)
{
    if (RefPtr download = m_downloads.get(downloadID))
        download->publishProgress(url, bookmarkData, useDownloadPlaceholder, activityAccessToken);
    else if (RefPtr pendingDownload = m_pendingDownloads.get(downloadID))
        pendingDownload->publishProgress(url, bookmarkData, useDownloadPlaceholder, activityAccessToken);
}
#else
void DownloadManager::publishDownloadProgress(DownloadID downloadID, const URL& url, SandboxExtension::Handle&& sandboxExtensionHandle)
{
    if (RefPtr download = m_downloads.get(downloadID))
        download->publishProgress(url, WTFMove(sandboxExtensionHandle));
    else if (RefPtr pendingDownload = m_pendingDownloads.get(downloadID))
        pendingDownload->publishProgress(url, WTFMove(sandboxExtensionHandle));
}
#endif
#endif // PLATFORM(COCOA)

void DownloadManager::downloadFinished(Download& download)
{
    ASSERT(m_downloads.get(download.downloadID()) == &download);
    download.clearManager();
    m_downloads.remove(download.downloadID());
}

void DownloadManager::didCreateDownload()
{
    m_client->didCreateDownload();
}

void DownloadManager::didDestroyDownload()
{
    m_client->didDestroyDownload();
}

IPC::Connection* DownloadManager::downloadProxyConnection()
{
    return m_client->downloadProxyConnection();
}

AuthenticationManager& DownloadManager::downloadsAuthenticationManager()
{
    return m_client->downloadsAuthenticationManager();
}

void DownloadManager::applicationDidEnterBackground()
{
    for (Ref download : m_downloads.values())
        download->applicationDidEnterBackground();
}

void DownloadManager::applicationWillEnterForeground()
{
    for (Ref download : m_downloads.values())
        download->applicationWillEnterForeground();
}

} // namespace WebKit
