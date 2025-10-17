/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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

#if ENABLE(REMOTE_INSPECTOR)

#include "RemoteInspector.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/SocketConnection.h>

typedef struct _GSocketAddress GSocketAddress;
typedef struct _GSocketConnection GSocketConnection;
typedef struct _GSocketService GSocketService;

namespace Inspector {

class RemoteInspectorServer {
public:
    JS_EXPORT_PRIVATE static RemoteInspectorServer& singleton();
    ~RemoteInspectorServer();

    JS_EXPORT_PRIVATE bool start(GRefPtr<GSocketAddress>&&);
    bool isRunning() const { return !!m_service; }
    uint16_t port() const { return m_port; }

private:
    static gboolean incomingConnectionCallback(GSocketService*, GSocketConnection*, GObject*, RemoteInspectorServer*);
    void incomingConnection(Ref<SocketConnection>&&);

    static const SocketConnection::MessageHandlers& messageHandlers();
    void connectionDidClose(SocketConnection&);
    void setTargetList(SocketConnection&, GVariant*);
    GVariant* setupInspectorClient(SocketConnection&, const char* clientBackendCommandsHash);
    void setup(SocketConnection&, uint64_t connectionID, uint64_t targetID);
    void close(SocketConnection&, uint64_t connectionID, uint64_t targetID);
    void sendMessageToFrontend(SocketConnection&, uint64_t target, const char*);
    void sendMessageToBackend(SocketConnection&, uint64_t connectionID, uint64_t targetID, const char*);
    void startAutomationSession(SocketConnection&, const char* sessionID, const RemoteInspector::Client::SessionCapabilities&);

    GRefPtr<GSocketService> m_service;
    uint16_t m_port { 0 };
    UncheckedKeyHashSet<RefPtr<SocketConnection>> m_connections;
    UncheckedKeyHashMap<SocketConnection*, uint64_t> m_remoteInspectorConnectionToIDMap;
    UncheckedKeyHashMap<uint64_t, SocketConnection*> m_idToRemoteInspectorConnectionMap;
    SocketConnection* m_clientConnection { nullptr };
    SocketConnection* m_automationConnection { nullptr };
    UncheckedKeyHashSet<std::pair<uint64_t, uint64_t>> m_inspectionTargets;
    UncheckedKeyHashSet<std::pair<uint64_t, uint64_t>> m_automationTargets;
};

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
