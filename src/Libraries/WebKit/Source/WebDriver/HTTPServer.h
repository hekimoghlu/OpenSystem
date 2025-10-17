/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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

#include <wtf/Forward.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

#if USE(SOUP)
#include <wtf/glib/GRefPtr.h>
typedef struct _SoupServer SoupServer;
#elif USE(INSPECTOR_SOCKET_SERVER)
#include "HTTPParser.h"
#include <JavaScriptCore/RemoteInspectorSocketEndpoint.h>

using Inspector::ConnectionID;
using Inspector::PlatformSocketType;
using Inspector::RemoteInspectorSocketEndpoint;
#endif

namespace WebDriver {

class HTTPRequestHandler
#if USE(INSPECTOR_SOCKET_SERVER)
: public RemoteInspectorSocketEndpoint::Client
#endif
{
public:
    struct Request {
        String method;
        String path;
        const char* data { nullptr };
        size_t dataLength { 0 };
    };
    struct Response {
        unsigned statusCode { 0 };
        CString data;
        String contentType;
    };
    virtual void handleRequest(Request&&, Function<void (Response&&)>&& replyHandler) = 0;

#if USE(INSPECTOR_SOCKET_SERVER)
    void connect(ConnectionID);
#endif

private:
#if USE(INSPECTOR_SOCKET_SERVER)
    void sendResponse(Response&&);
    void reset();
    String packHTTPMessage(Response&&) const;

    void didReceive(RemoteInspectorSocketEndpoint&, ConnectionID, Vector<uint8_t>&&) final;
    void didClose(RemoteInspectorSocketEndpoint&, ConnectionID) final;

    std::optional<ConnectionID> m_client;
    HTTPParser m_parser;
#endif
};

class HTTPServer
#if USE(INSPECTOR_SOCKET_SERVER)
: public RemoteInspectorSocketEndpoint::Listener
#endif
{
public:
    explicit HTTPServer(HTTPRequestHandler&);
    ~HTTPServer() = default;

    bool listen(const std::optional<String>& host, unsigned port);
    void disconnect();

    const String& visibleHost() const { return m_visibleHost; }

private:
#if USE(INSPECTOR_SOCKET_SERVER)
    std::optional<ConnectionID> doAccept(RemoteInspectorSocketEndpoint&, PlatformSocketType) final;
    void didChangeStatus(RemoteInspectorSocketEndpoint&, ConnectionID, RemoteInspectorSocketEndpoint::Listener::Status) final;
#endif

    HTTPRequestHandler& m_requestHandler;
    String m_visibleHost;

#if USE(SOUP)
    GRefPtr<SoupServer> m_soupServer;
#endif

#if USE(INSPECTOR_SOCKET_SERVER)
    std::optional<ConnectionID> m_server;
#endif
};

} // namespace WebDriver
