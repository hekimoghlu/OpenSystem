/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
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

#include "Connection.h"
#include "Download.h"
#include "NetworkConnectionToWebProcess.h"
#include "NetworkDataTask.h"
#include <WebCore/FetchIdentifier.h>
#include <wtf/FileSystem.h>
#include <wtf/FunctionDispatcher.h>
#include <wtf/TZoneMalloc.h>

namespace IPC {
class FormDataReference;
class SharedBufferReference;
}

namespace WebKit {

class NetworkLoad;
class NetworkProcess;
class SandboxExtension;
class WebSWServerToContextConnection;

class ServiceWorkerDownloadTask : public NetworkDataTask, private FunctionDispatcher, private IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(ServiceWorkerDownloadTask);
public:
    static Ref<ServiceWorkerDownloadTask> create(NetworkSession& session, NetworkDataTaskClient& client, WebSWServerToContextConnection& connection, WebCore::ServiceWorkerIdentifier serviceWorkerIdentifier, WebCore::SWServerConnectionIdentifier serverConnectionIdentifier, WebCore::FetchIdentifier fetchIdentifier, const WebCore::ResourceRequest& request, const WebCore::ResourceResponse& response, DownloadID downloadID)
    {
        auto task = adoptRef(*new ServiceWorkerDownloadTask(session, client, connection, serviceWorkerIdentifier, serverConnectionIdentifier, fetchIdentifier, request, response, downloadID));
        task->startListeningForIPC();
        return task;
    }
    ~ServiceWorkerDownloadTask();

    void ref() const final { NetworkDataTask::ref(); }
    void deref() const final { NetworkDataTask::deref(); }

    WebCore::FetchIdentifier fetchIdentifier() const { return m_fetchIdentifier; }
    void contextClosed() { cancel(); }
    void start();
    void stop() { cancel(); }

private:
    ServiceWorkerDownloadTask(NetworkSession&, NetworkDataTaskClient&, WebSWServerToContextConnection&, WebCore::ServiceWorkerIdentifier, WebCore::SWServerConnectionIdentifier, WebCore::FetchIdentifier, const WebCore::ResourceRequest&, const WebCore::ResourceResponse& response, DownloadID);
    void startListeningForIPC();

    Ref<NetworkProcess> protectedNetworkProcess() const;

    // IPC Message
    void didReceiveData(const IPC::SharedBufferReference&, uint64_t encodedDataLength);
    void didReceiveFormData(const IPC::FormDataReference&);
    void didFinish();
    void didFail(WebCore::ResourceError&&);

    // NetworkDataTask
    void cancel() final;
    void resume() final;
    void invalidateAndCancel() final;
    State state() const final { return m_state; }
    void setPendingDownloadLocation(const String& filename, SandboxExtension::Handle&&, bool /*allowOverwrite*/) final;

    // FunctionDispatcher
    void dispatch(Function<void()>&&) final;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    template<typename Message> bool sendToServiceWorker(Message&&);
    void didFailDownload(std::optional<WebCore::ResourceError>&& = { });
    void close();

    WeakPtr<WebSWServerToContextConnection> m_serviceWorkerConnection;
    WebCore::ServiceWorkerIdentifier m_serviceWorkerIdentifier;
    WebCore::SWServerConnectionIdentifier m_serverConnectionIdentifier;
    WebCore::FetchIdentifier m_fetchIdentifier;
    DownloadID m_downloadID;
    Ref<NetworkProcess> m_networkProcess;
    RefPtr<SandboxExtension> m_sandboxExtension;
    FileSystem::PlatformFileHandle m_downloadFile { FileSystem::invalidPlatformFileHandle };
    uint64_t m_downloadBytesWritten { 0 };
    std::optional<uint64_t> m_expectedContentLength;
    State m_state { State::Suspended };
};

}
