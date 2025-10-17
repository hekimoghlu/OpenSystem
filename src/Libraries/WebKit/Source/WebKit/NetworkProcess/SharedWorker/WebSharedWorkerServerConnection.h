/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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

#include "MessageReceiver.h"
#include "MessageSender.h"
#include <WebCore/ProcessIdentifier.h>
#include <WebCore/SharedWorkerObjectIdentifier.h>
#include <WebCore/TransferredMessagePort.h>
#include <WebCore/WorkerInitializationData.h>
#include <pal/SessionID.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {
class WebSharedWorkerServerConnection;
}

namespace WebCore {
class ResourceError;
struct SharedWorkerKey;
struct WorkerFetchResult;
struct WorkerOptions;
}

namespace WebKit {

class NetworkProcess;
class NetworkSession;
class WebSharedWorker;
class WebSharedWorkerServer;
class WebSharedWorkerServerToContextConnection;
class NetworkSession;

class WebSharedWorkerServerConnection : public IPC::MessageSender, public IPC::MessageReceiver, public RefCounted<WebSharedWorkerServerConnection> {
    WTF_MAKE_TZONE_ALLOCATED(WebSharedWorkerServerConnection);
public:
    static Ref<WebSharedWorkerServerConnection> create(NetworkProcess&, WebSharedWorkerServer&, IPC::Connection&, WebCore::ProcessIdentifier);

    ~WebSharedWorkerServerConnection();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    WebSharedWorkerServer* server();
    const WebSharedWorkerServer* server() const;

    NetworkSession* session();
    WebCore::ProcessIdentifier webProcessIdentifier() const { return m_webProcessIdentifier; }

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    void fetchScriptInClient(const WebSharedWorker&, WebCore::SharedWorkerObjectIdentifier, CompletionHandler<void(WebCore::WorkerFetchResult&&, WebCore::WorkerInitializationData&&)>&&);
    void notifyWorkerObjectOfLoadCompletion(WebCore::SharedWorkerObjectIdentifier, const WebCore::ResourceError&);
    void postErrorToWorkerObject(WebCore::SharedWorkerObjectIdentifier, const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL, bool isErrorEvent);

private:
    WebSharedWorkerServerConnection(NetworkProcess&, WebSharedWorkerServer&, IPC::Connection&, WebCore::ProcessIdentifier);
    Ref<NetworkProcess> protectedNetworkProcess();

    // IPC::MessageSender.
    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final { return 0; }

    // IPC messages.
    void requestSharedWorker(WebCore::SharedWorkerKey&&, WebCore::SharedWorkerObjectIdentifier, WebCore::TransferredMessagePort&&, WebCore::WorkerOptions&&);
    void sharedWorkerObjectIsGoingAway(WebCore::SharedWorkerKey&&, WebCore::SharedWorkerObjectIdentifier);
    void suspendForBackForwardCache(WebCore::SharedWorkerKey&&, WebCore::SharedWorkerObjectIdentifier);
    void resumeForBackForwardCache(WebCore::SharedWorkerKey&&, WebCore::SharedWorkerObjectIdentifier);

    Ref<IPC::Connection> m_contentConnection;
    Ref<NetworkProcess> m_networkProcess;
    WeakPtr<WebSharedWorkerServer> m_server;
    WebCore::ProcessIdentifier m_webProcessIdentifier;
};

} // namespace WebKit
