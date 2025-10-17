/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
#include "DownloadManager.h"
#include "DownloadMonitor.h"
#include "MessageSender.h"
#include "NetworkDataTask.h"
#include "SandboxExtension.h"
#include "UseDownloadPlaceholder.h"
#include <WebCore/AuthenticationChallenge.h>
#include <WebCore/ResourceRequest.h>
#include <memory>
#include <pal/SessionID.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

#if PLATFORM(COCOA)
OBJC_CLASS NSProgress;
OBJC_CLASS NSURLSessionDownloadTask;
#endif

namespace WebCore {
class AuthenticationChallenge;
class BlobDataFileReference;
class Credential;
class ResourceError;
class ResourceRequest;
class ResourceResponse;
}

namespace WebKit {

class DownloadMonitor;
class NetworkDataTask;
class NetworkSession;
class WebPage;

class Download final : public IPC::MessageSender, public RefCountedAndCanMakeWeakPtr<Download> {
    WTF_MAKE_TZONE_ALLOCATED(Download);
    WTF_MAKE_NONCOPYABLE(Download);
public:
    static Ref<Download> create(DownloadManager&, DownloadID, NetworkDataTask&, NetworkSession&, const String& suggestedFilename = { });
#if PLATFORM(COCOA)
    static Ref<Download> create(DownloadManager&, DownloadID, NSURLSessionDownloadTask*, NetworkSession&, const String& suggestedFilename = { });
#endif

    ~Download();

    void resume(std::span<const uint8_t> resumeData, const String& path, SandboxExtension::Handle&&, std::span<const uint8_t> activityAccessToken);
    enum class IgnoreDidFailCallback : bool { No, Yes };
    void cancel(CompletionHandler<void(std::span<const uint8_t>)>&&, IgnoreDidFailCallback);
#if PLATFORM(COCOA)
    void publishProgress(const URL&, SandboxExtension::Handle&&);
#endif

#if HAVE(MODERN_DOWNLOADPROGRESS)
    void publishProgress(const URL&, std::span<const uint8_t>, WebKit::UseDownloadPlaceholder, std::span<const uint8_t>);
    void setPlaceholderURL(NSURL *, NSData *);
    void setFinalURL(NSURL *, NSData *);
#endif

    DownloadID downloadID() const { return m_downloadID; }
    PAL::SessionID sessionID() const { return m_sessionID; }

    void setSandboxExtension(RefPtr<SandboxExtension>&& sandboxExtension) { m_sandboxExtension = WTFMove(sandboxExtension); }
    void didReceiveChallenge(const WebCore::AuthenticationChallenge&, ChallengeCompletionHandler&&);
    void didCreateDestination(const String& path);
    void didReceiveData(uint64_t bytesWritten, uint64_t totalBytesWritten, uint64_t totalBytesExpectedToWrite);
    void didFinish();
    void didFail(const WebCore::ResourceError&, std::span<const uint8_t> resumeData);

    void applicationDidEnterBackground() { protectedMonitor()->applicationDidEnterBackground(); }
    void applicationWillEnterForeground() { protectedMonitor()->applicationWillEnterForeground(); }
    DownloadManager* manager() const { return m_downloadManager.get(); }
    void clearManager() { m_downloadManager = nullptr; }

    unsigned testSpeedMultiplier() const { return m_testSpeedMultiplier; }

private:
    Download(DownloadManager&, DownloadID, NetworkDataTask&, NetworkSession&, const String& suggestedFilename = { });
#if PLATFORM(COCOA)
    Download(DownloadManager&, DownloadID, NSURLSessionDownloadTask*, NetworkSession&, const String& suggestedFilename = { });
#endif

    Ref<DownloadMonitor> protectedMonitor() { return m_monitor; }

    // IPC::MessageSender
    IPC::Connection* messageSenderConnection() const override;
    uint64_t messageSenderDestinationID() const override;

    void platformCancelNetworkLoad(CompletionHandler<void(std::span<const uint8_t>)>&&);
    void platformDestroyDownload();
    void platformDidFinish(CompletionHandler<void()>&&);

#if HAVE(MODERN_DOWNLOADPROGRESS)
    void startUpdatingProgress();
    void updateProgress(uint64_t totalBytesWritten, uint64_t totalBytesExpectedToWrite);
    static Vector<uint8_t> updateResumeDataWithPlaceholderURL(NSURL *, std::span<const uint8_t> resumeData);
#endif

    CheckedPtr<DownloadManager> m_downloadManager;
    DownloadID m_downloadID;
    Ref<DownloadManager::Client> m_client;

    Vector<RefPtr<WebCore::BlobDataFileReference>> m_blobFileReferences;
    RefPtr<SandboxExtension> m_sandboxExtension;

    RefPtr<NetworkDataTask> m_download;
#if PLATFORM(COCOA)
    RetainPtr<NSURLSessionDownloadTask> m_downloadTask;
    RetainPtr<NSProgress> m_progress;
    RetainPtr<NSURL> m_placeholderURL;
#if HAVE(MODERN_DOWNLOADPROGRESS)
    RetainPtr<NSData> m_bookmarkData;
    RetainPtr<NSURL> m_bookmarkURL;
    bool m_canUpdateProgress { false };
    std::optional<uint64_t> m_totalBytesWritten;
    std::optional<uint64_t> m_totalBytesExpectedToWrite;
#endif
#endif
    PAL::SessionID m_sessionID;
    bool m_hasReceivedData { false };
    IgnoreDidFailCallback m_ignoreDidFailCallback { IgnoreDidFailCallback::No };
    DownloadMonitor m_monitor { *this };
    unsigned m_testSpeedMultiplier { 1 };
    CompletionHandler<void(std::span<const uint8_t>)> m_cancelCompletionHandler;
};

} // namespace WebKit
