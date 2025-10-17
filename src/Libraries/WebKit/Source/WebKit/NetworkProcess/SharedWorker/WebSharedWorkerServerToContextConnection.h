/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#include "WebPageProxyIdentifier.h"
#include <WebCore/SharedWorkerIdentifier.h>
#include <WebCore/SharedWorkerObjectIdentifier.h>
#include <WebCore/Site.h>
#include <WebCore/Timer.h>
#include <WebCore/TransferredMessagePort.h>
#include <wtf/CheckedRef.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {
class WebSharedWorkerServerToContextConnection;
}

namespace WebCore {
class ScriptBuffer;
struct ClientOrigin;
struct WorkerFetchResult;
struct WorkerOptions;
}

namespace WebKit {

class NetworkConnectionToWebProcess;
class WebSharedWorker;
class WebSharedWorkerServer;

class WebSharedWorkerServerToContextConnection final : public IPC::MessageSender, public IPC::MessageReceiver, public RefCounted<WebSharedWorkerServerToContextConnection> {
    WTF_MAKE_TZONE_ALLOCATED(WebSharedWorkerServerToContextConnection);
public:
    static Ref<WebSharedWorkerServerToContextConnection> create(NetworkConnectionToWebProcess&, const WebCore::Site&, WebSharedWorkerServer&);

    ~WebSharedWorkerServerToContextConnection();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    std::optional<WebCore::ProcessIdentifier> webProcessIdentifier() const;
    const WebCore::RegistrableDomain& registrableDomain() const { return m_site.domain(); }
    const WebCore::Site& site() const { return m_site; }
    IPC::Connection* ipcConnection() const;

    void terminateWhenPossible() { m_shouldTerminateWhenPossible = true; }

    void launchSharedWorker(WebSharedWorker&);
    void postConnectEvent(const WebSharedWorker&, const WebCore::TransferredMessagePort&, CompletionHandler<void(bool)>&&);
    void terminateSharedWorker(const WebSharedWorker&);

    void suspendSharedWorker(WebCore::SharedWorkerIdentifier);
    void resumeSharedWorker(WebCore::SharedWorkerIdentifier);

    const HashMap<WebCore::ProcessIdentifier, HashSet<WebCore::SharedWorkerObjectIdentifier>>& sharedWorkerObjects() const { return m_sharedWorkerObjects; }

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    void addSharedWorkerObject(WebCore::SharedWorkerObjectIdentifier);
    void removeSharedWorkerObject(WebCore::SharedWorkerObjectIdentifier);

private:
    WebSharedWorkerServerToContextConnection(NetworkConnectionToWebProcess&, const WebCore::Site&, WebSharedWorkerServer&);

    void idleTerminationTimerFired();
    void connectionIsNoLongerNeeded();

    // IPC messages.
    void postErrorToWorkerObject(WebCore::SharedWorkerIdentifier, const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL, bool isErrorEvent);
    void sharedWorkerTerminated(WebCore::SharedWorkerIdentifier);

    // IPC::MessageSender.
    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final;

    WeakPtr<NetworkConnectionToWebProcess> m_connection;
    WeakPtr<WebSharedWorkerServer> m_server;
    WebCore::Site m_site;
    HashMap<WebCore::ProcessIdentifier, HashSet<WebCore::SharedWorkerObjectIdentifier>> m_sharedWorkerObjects;
    WebCore::Timer m_idleTerminationTimer;
    bool m_shouldTerminateWhenPossible { false };
};

} // namespace WebKit
