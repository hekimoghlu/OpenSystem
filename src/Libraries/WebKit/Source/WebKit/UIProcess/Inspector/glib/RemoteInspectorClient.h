/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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

#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/SocketConnection.h>
#include <wtf/text/WTFString.h>

typedef struct _GCancellable GCancellable;

namespace WebKit {

class RemoteInspectorClient;
class RemoteInspectorProxy;
class RemoteWebInspectorUIProxy;

class RemoteInspectorObserver {
public:
    virtual ~RemoteInspectorObserver() { }
    virtual void targetListChanged(RemoteInspectorClient&) = 0;
    virtual void connectionClosed(RemoteInspectorClient&) = 0;
};

class RemoteInspectorClient {
    WTF_MAKE_TZONE_ALLOCATED(RemoteInspectorClient);
public:
    RemoteInspectorClient(String&& hostAndPort, RemoteInspectorObserver&);
    ~RemoteInspectorClient();

    const String& hostAndPort() const { return m_hostAndPort; }
    const String& backendCommandsURL() const { return m_backendCommandsURL; }

    enum class InspectorType { UI, HTTP };
    GString* buildTargetListPage(InspectorType) const;
    enum class ShouldEscapeSingleQuote : bool { No, Yes };
    void appendTargertList(GString*, InspectorType, ShouldEscapeSingleQuote) const;
    void inspect(uint64_t connectionID, uint64_t targetID, const String& targetType, InspectorType = InspectorType::UI);
    void sendMessageToBackend(uint64_t connectionID, uint64_t targetID, const String&);
    void closeFromFrontend(uint64_t connectionID, uint64_t targetID);

private:
    static const SocketConnection::MessageHandlers& messageHandlers();
    void setupConnection(Ref<SocketConnection>&&);
    void connectionDidClose();

    struct Target {
        uint64_t id;
        CString type;
        CString name;
        CString url;
    };

    void setBackendCommands(const char*);
    void setTargetList(uint64_t connectionID, Vector<Target>&&);
    void sendMessageToFrontend(uint64_t connectionID, uint64_t targetID, const char*);

    String m_hostAndPort;
    String m_backendCommandsURL;
    RemoteInspectorObserver& m_observer;
    RefPtr<SocketConnection> m_socketConnection;
    GRefPtr<GCancellable> m_cancellable;
    HashMap<uint64_t, Vector<Target>> m_targets;
    HashMap<std::pair<uint64_t, uint64_t>, std::unique_ptr<RemoteInspectorProxy>> m_inspectorProxyMap;
};

} // namespace WebKit

#endif // ENABLE(REMOTE_INSPECTOR)
