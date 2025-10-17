/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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

#include "RemoteInspectorClient.h"
#include <libsoup/soup.h>
#include <wtf/glib/GRefPtr.h>

namespace WebKit {

class RemoteInspectorHTTPServer final : public RemoteInspectorObserver {
public:
    static RemoteInspectorHTTPServer& singleton();
    ~RemoteInspectorHTTPServer() = default;

    bool start(GRefPtr<GSocketAddress>&&, unsigned inspectorPort);
    bool isRunning() const { return !!m_server; }
    const String& inspectorServerAddress() const;

    void sendMessageToFrontend(uint64_t connectionID, uint64_t targetID, const String& message) const;
    void targetDidClose(uint64_t connectionID, uint64_t targetID);

private:
    unsigned handleRequest(const char*, SoupMessageHeaders*, SoupMessageBody*) const;
    void handleWebSocket(const char*, SoupWebsocketConnection*);

    void sendMessageToBackend(SoupWebsocketConnection*, const String&) const;
    void didCloseWebSocket(SoupWebsocketConnection*);

    void targetListChanged(RemoteInspectorClient&) override { }
    void connectionClosed(RemoteInspectorClient&) override { }

    GRefPtr<SoupServer> m_server;
    std::unique_ptr<RemoteInspectorClient> m_client;
    HashMap<std::pair<uint64_t, uint64_t>, GRefPtr<SoupWebsocketConnection>> m_webSocketConnectionMap;
    HashMap<SoupWebsocketConnection*, std::pair<uint64_t, uint64_t>> m_webSocketConnectionToTargetMap;
};

} // namespace WebKit

#endif // ENABLE(REMOTE_INSPECTOR)
