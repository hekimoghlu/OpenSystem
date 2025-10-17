/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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

#include <WebCore/ProcessIdentifier.h>
#include <WebCore/RegistrableDomain.h>
#include <WebCore/ScriptExecutionContextIdentifier.h>
#include <WebCore/SharedWorkerIdentifier.h>
#include <WebCore/SharedWorkerKey.h>
#include <WebCore/SharedWorkerObjectIdentifier.h>
#include <WebCore/TransferredMessagePort.h>
#include <wtf/CheckedPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebKit {
class WebSharedWorkerServer;
}

namespace PAL {
class SessionID;
}

namespace WebCore {
class Site;
struct ClientOrigin;
struct WorkerFetchResult;
struct WorkerInitializationData;
struct WorkerOptions;
}

namespace WebKit {

class NetworkSession;
class WebSharedWorker;
class WebSharedWorkerServerConnection;
class WebSharedWorkerServerToContextConnection;

class WebSharedWorkerServer : public CanMakeWeakPtr<WebSharedWorkerServer>, public CanMakeCheckedPtr<WebSharedWorkerServer> {
    WTF_MAKE_TZONE_ALLOCATED(WebSharedWorkerServer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebSharedWorkerServer);
public:
    explicit WebSharedWorkerServer(NetworkSession&);
    ~WebSharedWorkerServer();

    PAL::SessionID sessionID();
    WebSharedWorkerServerToContextConnection* contextConnectionForRegistrableDomain(const WebCore::RegistrableDomain&) const;

    void requestSharedWorker(WebCore::SharedWorkerKey&&, WebCore::SharedWorkerObjectIdentifier, WebCore::TransferredMessagePort&&, WebCore::WorkerOptions&&);
    void sharedWorkerObjectIsGoingAway(const WebCore::SharedWorkerKey&, WebCore::SharedWorkerObjectIdentifier);
    void suspendForBackForwardCache(const WebCore::SharedWorkerKey&, WebCore::SharedWorkerObjectIdentifier);
    void resumeForBackForwardCache(const WebCore::SharedWorkerKey&, WebCore::SharedWorkerObjectIdentifier);
    void postErrorToWorkerObject(WebCore::SharedWorkerIdentifier, const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL, bool isErrorEvent);
    void sharedWorkerTerminated(WebCore::SharedWorkerIdentifier);

    void terminateContextConnectionWhenPossible(const WebCore::RegistrableDomain&, WebCore::ProcessIdentifier);
    void addContextConnection(WebSharedWorkerServerToContextConnection&);
    void removeContextConnection(WebSharedWorkerServerToContextConnection&);
    void addConnection(Ref<WebSharedWorkerServerConnection>&&);
    void removeConnection(WebCore::ProcessIdentifier);

private:
    void createContextConnection(const WebCore::Site&, std::optional<WebCore::ProcessIdentifier> requestingProcessIdentifier);
    bool needsContextConnectionForRegistrableDomain(const WebCore::RegistrableDomain&) const;
    void contextConnectionCreated(WebSharedWorkerServerToContextConnection&);
    void didFinishFetchingSharedWorkerScript(WebSharedWorker&, WebCore::WorkerFetchResult&&, WebCore::WorkerInitializationData&&);
    void shutDownSharedWorker(const WebCore::SharedWorkerKey&);

    CheckedRef<NetworkSession> m_session;
    HashMap<WebCore::ProcessIdentifier, Ref<WebSharedWorkerServerConnection>> m_connections;
    HashMap<WebCore::RegistrableDomain, WeakRef<WebSharedWorkerServerToContextConnection>> m_contextConnections;
    HashSet<WebCore::RegistrableDomain> m_pendingContextConnectionDomains;
    HashMap<WebCore::SharedWorkerKey, Ref<WebSharedWorker>> m_sharedWorkers;
};

} // namespace WebKit
