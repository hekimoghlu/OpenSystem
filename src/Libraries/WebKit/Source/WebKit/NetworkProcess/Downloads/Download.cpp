/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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
#include "Download.h"

#include "AuthenticationChallengeDisposition.h"
#include "AuthenticationManager.h"
#include "Connection.h"
#include "DownloadManager.h"
#include "DownloadMonitor.h"
#include "DownloadProxyMessages.h"
#include "Logging.h"
#include "MessageSenderInlines.h"
#include "NetworkDataTask.h"
#include "NetworkProcess.h"
#include "NetworkSession.h"
#include "SandboxExtension.h"
#include <WebCore/NotImplemented.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)
#include "NetworkDataTaskCocoa.h"
#endif

#define DOWNLOAD_RELEASE_LOG(fmt, ...) RELEASE_LOG(Network, "%p - Download::" fmt, this, ##__VA_ARGS__)

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(Download);

Ref<Download> Download::create(DownloadManager& downloadManager, DownloadID downloadID, NetworkDataTask& download, NetworkSession& session, const String& suggestedName)
{
    return adoptRef(*new Download(downloadManager, downloadID, download, session, suggestedName));
}

Download::Download(DownloadManager& downloadManager, DownloadID downloadID, NetworkDataTask& download, NetworkSession& session, const String& suggestedName)
    : m_downloadManager(downloadManager)
    , m_downloadID(downloadID)
    , m_client(downloadManager.client())
    , m_download(&download)
    , m_sessionID(session.sessionID())
    , m_testSpeedMultiplier(session.testSpeedMultiplier())
{
    downloadManager.didCreateDownload();
}

#if PLATFORM(COCOA)
Ref<Download> Download::create(DownloadManager& downloadManager, DownloadID downloadID, NSURLSessionDownloadTask* download, NetworkSession& session, const String& suggestedName)
{
    return adoptRef(*new Download(downloadManager, downloadID, download, session, suggestedName));
}

Download::Download(DownloadManager& downloadManager, DownloadID downloadID, NSURLSessionDownloadTask* download, NetworkSession& session, const String& suggestedName)
    : m_downloadManager(downloadManager)
    , m_downloadID(downloadID)
    , m_client(downloadManager.client())
    , m_downloadTask(download)
    , m_sessionID(session.sessionID())
    , m_testSpeedMultiplier(session.testSpeedMultiplier())
{
    downloadManager.didCreateDownload();
}
#endif

Download::~Download()
{
    platformDestroyDownload();
    if (CheckedPtr downloadManager = m_downloadManager)
        downloadManager->didDestroyDownload();
}

void Download::cancel(CompletionHandler<void(std::span<const uint8_t>)>&& completionHandler, IgnoreDidFailCallback ignoreDidFailCallback)
{
    RELEASE_ASSERT(isMainRunLoop());

    // URLSession:task:didCompleteWithError: is still called after cancelByProducingResumeData's completionHandler.
    // If this cancel request came from the API, we do not want to send DownloadProxy::DidFail because the
    // completionHandler will inform the API that the cancellation succeeded.
    m_ignoreDidFailCallback = ignoreDidFailCallback;

    auto completionHandlerWrapper = [this, weakThis = WeakPtr { *this }, completionHandler = WTFMove(completionHandler)] (std::span<const uint8_t> resumeData) mutable {
        completionHandler(resumeData);
        if (!weakThis || m_ignoreDidFailCallback == IgnoreDidFailCallback::No)
            return;
        DOWNLOAD_RELEASE_LOG("didCancel: (id = %" PRIu64 ")", downloadID().toUInt64());
        if (auto extension = std::exchange(m_sandboxExtension, nullptr))
            extension->revoke();
        if (CheckedPtr downloadManager = m_downloadManager)
            downloadManager->downloadFinished(*this);
    };

    if (m_download) {
        m_download->cancel();
        completionHandlerWrapper({ });
        return;
    }
    platformCancelNetworkLoad(WTFMove(completionHandlerWrapper));
}

void Download::didReceiveChallenge(const WebCore::AuthenticationChallenge& challenge, ChallengeCompletionHandler&& completionHandler)
{
    if (challenge.protectionSpace().isPasswordBased() && !challenge.proposedCredential().isEmpty() && !challenge.previousFailureCount()) {
        completionHandler(AuthenticationChallengeDisposition::UseCredential, challenge.proposedCredential());
        return;
    }

    m_client->downloadsAuthenticationManager().didReceiveAuthenticationChallenge(*this, challenge, WTFMove(completionHandler));
}

void Download::didCreateDestination(const String& path)
{
    send(Messages::DownloadProxy::DidCreateDestination(path));
}

void Download::didReceiveData(uint64_t bytesWritten, uint64_t totalBytesWritten, uint64_t totalBytesExpectedToWrite)
{
    if (!m_hasReceivedData) {
        DOWNLOAD_RELEASE_LOG("didReceiveData: Started receiving data (id = %" PRIu64 ", expected length = %" PRIu64 ")", downloadID().toUInt64(), totalBytesExpectedToWrite);
        m_hasReceivedData = true;
    }
    
    m_monitor.downloadReceivedBytes(bytesWritten);

#if HAVE(MODERN_DOWNLOADPROGRESS)
    updateProgress(totalBytesWritten, totalBytesExpectedToWrite);
#endif

    send(Messages::DownloadProxy::DidReceiveData(bytesWritten, totalBytesWritten, totalBytesExpectedToWrite));
}

void Download::didFinish()
{
    DOWNLOAD_RELEASE_LOG("didFinish: (id = %" PRIu64 ")", downloadID().toUInt64());

    platformDidFinish([weakThis = WeakPtr { *this }, this] {
        RELEASE_ASSERT(isMainRunLoop());
        if (!weakThis)
            return;
        send(Messages::DownloadProxy::DidFinish());

        if (m_sandboxExtension) {
            m_sandboxExtension->revoke();
            m_sandboxExtension = nullptr;
        }

        if (CheckedPtr downloadManager = m_downloadManager)
            downloadManager->downloadFinished(*this);
    });
}

void Download::didFail(const ResourceError& error, std::span<const uint8_t> resumeData)
{
    if (m_ignoreDidFailCallback == IgnoreDidFailCallback::Yes)
        return;

    DOWNLOAD_RELEASE_LOG("didFail: (id = %" PRIu64 ", isTimeout = %d, isCancellation = %d, errCode = %d)",
        downloadID().toUInt64(), error.isTimeout(), error.isCancellation(), error.errorCode());

#if HAVE(MODERN_DOWNLOADPROGRESS)
    auto resumeDataWithPlaceholder = updateResumeDataWithPlaceholderURL(m_placeholderURL.get(), resumeData);
    resumeData = resumeDataWithPlaceholder.span();
#endif

    send(Messages::DownloadProxy::DidFail(error, resumeData));

    if (m_sandboxExtension) {
        m_sandboxExtension->revoke();
        m_sandboxExtension = nullptr;
    }
    if (CheckedPtr downloadManager = m_downloadManager)
        downloadManager->downloadFinished(*this);
}

IPC::Connection* Download::messageSenderConnection() const
{
    if (CheckedPtr downloadManager = m_downloadManager)
        return downloadManager->downloadProxyConnection();
    return nullptr;
}

uint64_t Download::messageSenderDestinationID() const
{
    return m_downloadID.toUInt64();
}

#if !PLATFORM(COCOA)
void Download::platformCancelNetworkLoad(CompletionHandler<void(std::span<const uint8_t>)>&& completionHandler)
{
    completionHandler({ });
}

void Download::platformDestroyDownload()
{
}

void Download::platformDidFinish(CompletionHandler<void()>&& completionHandler)
{
    completionHandler();
}
#endif

} // namespace WebKit

#undef DOWNLOAD_RELEASE_LOG
