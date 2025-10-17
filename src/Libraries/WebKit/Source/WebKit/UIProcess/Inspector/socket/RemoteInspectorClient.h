/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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

#include <JavaScriptCore/RemoteControllableTarget.h>
#include <JavaScriptCore/RemoteInspectorConnectionClient.h>
#include <WebCore/InspectorDebuggableType.h>
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class RemoteInspectorClient;
class RemoteInspectorProxy;

class RemoteInspectorObserver : public CanMakeCheckedPtr<RemoteInspectorObserver> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteInspectorObserver);
public:
    virtual ~RemoteInspectorObserver() { }
    virtual void targetListChanged(RemoteInspectorClient&) = 0;
    virtual void connectionClosed(RemoteInspectorClient&) = 0;
};

using ConnectionID = Inspector::ConnectionID;
using TargetID = Inspector::TargetID;

class RemoteInspectorClient final : public Inspector::RemoteInspectorConnectionClient, public CanMakeCheckedPtr<RemoteInspectorClient> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteInspectorClient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteInspectorClient);
public:
    RemoteInspectorClient(URL, RemoteInspectorObserver&);
    ~RemoteInspectorClient();

    // FIXME: We should update the messaging protocol to use DebuggableType instead of String for the target type.
    // https://bugs.webkit.org/show_bug.cgi?id=206047
    struct Target {
        TargetID id;
        String type;
        String name;
        String url;
    };

    const HashMap<ConnectionID, Vector<Target>>& targets() const { return m_targets; }
    const String& backendCommandsURL() const { return m_backendCommandsURL; }

    void inspect(ConnectionID, TargetID, Inspector::DebuggableType);
    void sendMessageToBackend(ConnectionID, TargetID, const String&);
    void closeFromFrontend(ConnectionID, TargetID);

private:
    friend class NeverDestroyed<RemoteInspectorClient>;

    void startInitialCommunication();
    void connectionClosed();

    void setTargetList(const Event&);
    void sendMessageToFrontend(const Event&);
    void setBackendCommands(const Event&);

    void didClose(Inspector::RemoteInspectorSocketEndpoint&, ConnectionID) final;
    HashMap<String, CallHandler>& dispatchMap() final;

    void sendWebInspectorEvent(const String&);

    String m_backendCommandsURL;
    CheckedRef<RemoteInspectorObserver> m_observer;
    std::optional<ConnectionID> m_connectionID;
    HashMap<ConnectionID, Vector<Target>> m_targets;
    HashMap<std::pair<ConnectionID, TargetID>, std::unique_ptr<RemoteInspectorProxy>> m_inspectorProxyMap;
};

} // namespace WebKit

#endif // ENABLE(REMOTE_INSPECTOR)
