/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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
#include "ServiceWorkerDownloadTask.h"

#include "DownloadManager.h"
#include "Logging.h"
#include "NetworkProcess.h"
#include "ServiceWorkerDownloadTaskMessages.h"
#include "SharedBufferReference.h"
#include "WebErrors.h"
#include "WebSWContextManagerConnectionMessages.h"
#include "WebSWServerToContextConnection.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/WorkQueue.h>

namespace WebKit {

using namespace WebCore;

static WorkQueue& serviceWorkerDownloadTaskQueueSingleton()
{
    static NeverDestroyed<Ref<WorkQueue>> queue(WorkQueue::create("Shared ServiceWorkerDownloadTask Queue"_s));
    return queue.get();
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(ServiceWorkerDownloadTask);

ServiceWorkerDownloadTask::ServiceWorkerDownloadTask(NetworkSession& session, NetworkDataTaskClient& client, WebSWServerToContextConnection& serviceWorkerConnection, ServiceWorkerIdentifier serviceWorkerIdentifier, SWServerConnectionIdentifier serverConnectionIdentifier, FetchIdentifier fetchIdentifier, const WebCore::ResourceRequest& request, const ResourceResponse& response, DownloadID downloadID)
    : NetworkDataTask(session, client, request, StoredCredentialsPolicy::DoNotUse, false, false)
    , m_serviceWorkerConnection(serviceWorkerConnection)
    , m_serviceWorkerIdentifier(serviceWorkerIdentifier)
    , m_serverConnectionIdentifier(serverConnectionIdentifier)
    , m_fetchIdentifier(fetchIdentifier)
    , m_downloadID(downloadID)
    , m_networkProcess(*serviceWorkerConnection.networkProcess())
{
    auto expectedContentLength = response.expectedContentLength();
    if (expectedContentLength != -1)
        m_expectedContentLength = expectedContentLength;
    serviceWorkerConnection.registerDownload(*this);
}

ServiceWorkerDownloadTask::~ServiceWorkerDownloadTask()
{
    ASSERT(!m_serviceWorkerConnection);
}

void ServiceWorkerDownloadTask::startListeningForIPC()
{
    RefPtr { m_serviceWorkerConnection.get() }->protectedIPCConnection()->addMessageReceiver(*this, *this, Messages::ServiceWorkerDownloadTask::messageReceiverName(), fetchIdentifier().toUInt64());
}

Ref<NetworkProcess> ServiceWorkerDownloadTask::protectedNetworkProcess() const
{
    return m_networkProcess;
}

void ServiceWorkerDownloadTask::close()
{
    ASSERT(isMainRunLoop());

    if (RefPtr serviceWorkerConnection = m_serviceWorkerConnection.get()) {
        serviceWorkerConnection->protectedIPCConnection()->removeMessageReceiver(Messages::ServiceWorkerDownloadTask::messageReceiverName(), fetchIdentifier().toUInt64());
        serviceWorkerConnection->unregisterDownload(*this);
        m_serviceWorkerConnection = nullptr;
    }
}

template<typename Message> bool ServiceWorkerDownloadTask::sendToServiceWorker(Message&& message)
{
    RefPtr serviceWorkerConnection = m_serviceWorkerConnection.get();
    if (!serviceWorkerConnection)
        return false;

    return serviceWorkerConnection->protectedIPCConnection()->send(std::forward<Message>(message), 0) == IPC::Error::NoError;
}

void ServiceWorkerDownloadTask::dispatch(Function<void()>&& function)
{
    serviceWorkerDownloadTaskQueueSingleton().dispatch([protectedThis = Ref { *this }, function = WTFMove(function)] {
        function();
    });
}

void ServiceWorkerDownloadTask::cancel()
{
    ASSERT(isMainRunLoop());

    serviceWorkerDownloadTaskQueueSingleton().dispatch([this, protectedThis = Ref { *this }] {
        if (m_downloadFile != FileSystem::invalidPlatformFileHandle) {
            FileSystem::closeFile(m_downloadFile);
            m_downloadFile = FileSystem::invalidPlatformFileHandle;
        }
    });

    if (RefPtr sandboxExtension = std::exchange(m_sandboxExtension, nullptr))
        sandboxExtension->revoke();

    sendToServiceWorker(Messages::WebSWContextManagerConnection::CancelFetch { m_serverConnectionIdentifier, m_serviceWorkerIdentifier, m_fetchIdentifier });

    m_state = State::Completed;
    close();
}

void ServiceWorkerDownloadTask::resume()
{
    ASSERT(isMainRunLoop());

    m_state = State::Running;
}

void ServiceWorkerDownloadTask::invalidateAndCancel()
{
    ASSERT(isMainRunLoop());

    cancel();
}

void ServiceWorkerDownloadTask::setPendingDownloadLocation(const WTF::String& filename, SandboxExtension::Handle&& sandboxExtensionHandle, bool allowOverwrite)
{
    ASSERT(isMainRunLoop());

    if (!networkSession()) {
        serviceWorkerDownloadTaskQueueSingleton().dispatch([this, protectedThis = Ref { *this }]() mutable {
            didFailDownload();
        });
        return;
    }

    NetworkDataTask::setPendingDownloadLocation(filename, { }, allowOverwrite);

    ASSERT(!m_sandboxExtension);
    m_sandboxExtension = SandboxExtension::create(WTFMove(sandboxExtensionHandle));
    if (RefPtr sandboxExtension = m_sandboxExtension)
        sandboxExtension->consume();

    serviceWorkerDownloadTaskQueueSingleton().dispatch([this, protectedThis = Ref { *this }, allowOverwrite, filename = filename.isolatedCopy()]() mutable {
        if (allowOverwrite && FileSystem::fileExists(filename)) {
            if (!FileSystem::deleteFile(filename)) {
                didFailDownload();
                return;
            }
        }

        m_downloadFile = FileSystem::openFile(m_pendingDownloadLocation, FileSystem::FileOpenMode::Truncate);
        if (m_downloadFile == FileSystem::invalidPlatformFileHandle)
            didFailDownload();
    });
}

void ServiceWorkerDownloadTask::start()
{
    ASSERT(m_state != State::Completed);

    if (!sendToServiceWorker(Messages::WebSWContextManagerConnection::ConvertFetchToDownload { m_serverConnectionIdentifier, m_serviceWorkerIdentifier, m_fetchIdentifier })) {
        serviceWorkerDownloadTaskQueueSingleton().dispatch([this, protectedThis = Ref { *this }]() mutable {
            didFailDownload();
        });
        return;
    }

    m_state = State::Running;

    auto& manager = protectedNetworkProcess()->downloadManager();
    Ref download = Download::create(manager, m_downloadID, *this, *networkSession());
    manager.dataTaskBecameDownloadTask(m_downloadID, download.copyRef());
    download->didCreateDestination(m_pendingDownloadLocation);
}

void ServiceWorkerDownloadTask::didReceiveData(const IPC::SharedBufferReference& data, uint64_t encodedDataLength)
{
    ASSERT(!isMainRunLoop());

    if (m_downloadFile == FileSystem::invalidPlatformFileHandle)
        return;

    size_t bytesWritten = FileSystem::writeToFile(m_downloadFile, data.span());

    if (bytesWritten != data.size()) {
        didFailDownload();
        return;
    }

    callOnMainRunLoop([this, protectedThis = Ref { *this }, bytesWritten] {
        m_downloadBytesWritten += bytesWritten;
        if (RefPtr download = protectedNetworkProcess()->downloadManager().download(*m_pendingDownloadID))
            download->didReceiveData(bytesWritten, m_downloadBytesWritten, std::max(m_expectedContentLength.value_or(0), m_downloadBytesWritten));
    });
}

void ServiceWorkerDownloadTask::didReceiveFormData(const IPC::FormDataReference& formData)
{
    ASSERT(!isMainRunLoop());

    // FIXME: Support writing formData in downloads.
    RELEASE_LOG_ERROR(ServiceWorker, "ServiceWorkerDownloadTask::didReceiveFormData not implemented");
    didFailDownload();
}

void ServiceWorkerDownloadTask::didFinish()
{
    ASSERT(!isMainRunLoop());

    FileSystem::closeFile(m_downloadFile);
    m_downloadFile = FileSystem::invalidPlatformFileHandle;

    callOnMainRunLoop([this, protectedThis = Ref { *this }] {
        m_state = State::Completed;
        close();

#if !HAVE(MODERN_DOWNLOADPROGRESS)
        if (RefPtr sandboxExtension = std::exchange(m_sandboxExtension, nullptr))
            sandboxExtension->revoke();
#endif

        if (RefPtr download = protectedNetworkProcess()->downloadManager().download(*m_pendingDownloadID)) {
#if HAVE(MODERN_DOWNLOADPROGRESS)
            if (RefPtr sandboxExtension = std::exchange(m_sandboxExtension, nullptr))
                download->setSandboxExtension(WTFMove(sandboxExtension));
#endif
            download->didFinish();
        }

        if (RefPtr client = m_client.get())
            client->didCompleteWithError({ });
    });
}

void ServiceWorkerDownloadTask::didFail(ResourceError&& error)
{
    ASSERT(!isMainRunLoop());

    didFailDownload(WTFMove(error));
}

void ServiceWorkerDownloadTask::didFailDownload(std::optional<ResourceError>&& error)
{
    ASSERT(!isMainRunLoop());

    if (m_downloadFile != FileSystem::invalidPlatformFileHandle) {
        FileSystem::closeFile(m_downloadFile);
        m_downloadFile = FileSystem::invalidPlatformFileHandle;
    }

    callOnMainRunLoop([this, protectedThis = Ref { *this }, error = crossThreadCopy(WTFMove(error))] {
        if (m_state == State::Completed)
            return;

        m_state = State::Completed;
        close();

        if (RefPtr sandboxExtension = std::exchange(m_sandboxExtension, nullptr))
            sandboxExtension->revoke();

        auto resourceError = error.value_or(cancelledError(firstRequest()));
        if (RefPtr download = protectedNetworkProcess()->downloadManager().download(*m_pendingDownloadID))
            download->didFail(resourceError, { });

        if (RefPtr client = m_client.get())
            client->didCompleteWithError(resourceError);
    });
}

} // namespace WebKit
