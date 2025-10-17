/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
#include "DownloadMap.h"
#include "NetworkDataTask.h"
#include "PendingDownload.h"
#include "PolicyDecision.h"
#include "SandboxExtension.h"
#include "UseDownloadPlaceholder.h"
#include <WebCore/LocalFrameLoaderClient.h>
#include <WebCore/NotImplemented.h>
#include <wtf/AbstractRefCounted.h>
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>

namespace PAL {
class SessionID;
}

namespace WebCore {
class BlobDataFileReference;
class ResourceRequest;
class ResourceResponse;

enum class FromDownloadAttribute : bool;
}

namespace IPC {
class Connection;
}

namespace WebKit {

class AuthenticationManager;
class Download;
class NetworkConnectionToWebProcess;
class NetworkLoad;
class PendingDownload;

enum class CallDownloadDidStart : bool { No, Yes };

class DownloadManager : public CanMakeCheckedPtr<DownloadManager> {
    WTF_MAKE_NONCOPYABLE(DownloadManager);
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DownloadManager);
public:
    class Client : public AbstractRefCounted {
    public:
        virtual ~Client() { }

        // CheckedPtr interface
        virtual uint32_t checkedPtrCount() const = 0;
        virtual uint32_t checkedPtrCountWithoutThreadCheck() const = 0;
        virtual void incrementCheckedPtrCount() const = 0;
        virtual void decrementCheckedPtrCount() const = 0;

        virtual void didCreateDownload() = 0;
        virtual void didDestroyDownload() = 0;
        virtual IPC::Connection* downloadProxyConnection() = 0;
        virtual IPC::Connection* parentProcessConnectionForDownloads() = 0;
        virtual AuthenticationManager& downloadsAuthenticationManager() = 0;
        virtual NetworkSession* networkSession(PAL::SessionID) const = 0;
    };

    explicit DownloadManager(Client&);
    ~DownloadManager();

    void startDownload(PAL::SessionID, DownloadID, const WebCore::ResourceRequest&, const std::optional<WebCore::SecurityOriginData>& topOrigin, std::optional<NavigatingToAppBoundDomain>, const String& suggestedName = { }, WebCore::FromDownloadAttribute = WebCore::FromDownloadAttribute::No, std::optional<WebCore::FrameIdentifier> frameID = std::nullopt, std::optional<WebCore::PageIdentifier> = std::nullopt, std::optional<WebCore::ProcessIdentifier> = std::nullopt);
    void dataTaskBecameDownloadTask(DownloadID, Ref<Download>&&);
    void convertNetworkLoadToDownload(DownloadID, Ref<NetworkLoad>&&, ResponseCompletionHandler&&,  Vector<RefPtr<WebCore::BlobDataFileReference>>&&, const WebCore::ResourceRequest&, const WebCore::ResourceResponse&);
    void downloadDestinationDecided(DownloadID, Ref<NetworkDataTask>&&);

    void resumeDownload(PAL::SessionID, DownloadID, std::span<const uint8_t> resumeData, const String& path, SandboxExtension::Handle&&, CallDownloadDidStart, std::span<const uint8_t> activityAccessToken);

    void cancelDownload(DownloadID, CompletionHandler<void(std::span<const uint8_t>)>&&);
#if PLATFORM(COCOA)
#if HAVE(MODERN_DOWNLOADPROGRESS)
    void publishDownloadProgress(DownloadID, const URL&, std::span<const uint8_t> bookmarkData, WebKit::UseDownloadPlaceholder, std::span<const uint8_t> activityAccessToken);
#else
    void publishDownloadProgress(DownloadID, const URL&, SandboxExtension::Handle&&);
#endif
#endif
    
    Download* download(DownloadID);

    void downloadFinished(Download&);
    bool isDownloading() const { return !m_downloads.isEmpty(); }

    void applicationDidEnterBackground();
    void applicationWillEnterForeground();

    void didCreateDownload();
    void didDestroyDownload();

    IPC::Connection* downloadProxyConnection();
    AuthenticationManager& downloadsAuthenticationManager();
    
    Client& client() { return m_client; }

private:
    CheckedRef<Client> m_client;
    HashMap<DownloadID, Ref<PendingDownload>> m_pendingDownloads;
    HashMap<DownloadID, RefPtr<NetworkDataTask>> m_downloadsAfterDestinationDecided;
    DownloadMap m_downloads;
};

} // namespace WebKit
