/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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

#include "CommandResult.h"
#include "HTTPServer.h"
#include <map>
#include <optional>
#include <vector>
#include <wtf/HashMap.h>
#include <wtf/JSONValues.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

#if USE(SOUP)
#include <wtf/glib/GRefPtr.h>
typedef struct _SoupServer SoupServer;
typedef struct _SoupWebsocketConnection SoupWebsocketConnection;
#endif

namespace WebDriver {

class Session;
class WebDriverService;

struct Command {
    // Defined as js-uint, but in practice opaque for the remote end.
    String id;
    String method;
    RefPtr<JSON::Object> parameters;

    static std::optional<Command> fromData(const char* data, size_t dataLength);
};

// https://w3c.github.io/webdriver-bidi/#websocket-listener
// A WebSocket listener is a network endpoint that is able to accept incoming WebSocket connections.
struct WebSocketListener : public RefCounted<WebSocketListener> {
    static RefPtr<WebSocketListener> create(const String& host, unsigned port, bool secure, const std::vector<String>& resources = { })
    {
        return adoptRef(*new WebSocketListener(host, port, secure, resources));
    }

    String host;
    unsigned port { 0 };
    bool isSecure { false };
    std::vector<String> resources { };

private:
    WebSocketListener(const String& host, unsigned port, bool isSecure, const std::vector<String>& resources)
        : host(host)
        , port(port)
        , isSecure(isSecure)
        , resources(resources)
    {
    }
};


class WebSocketMessageHandler {
public:

#if USE(SOUP)
    using Connection = GRefPtr<SoupWebsocketConnection>;
#endif

    struct Message {
        // Optional connection, as the message might be generated without a connection object available (e.g. inside a method handler).
        // In this case, it gets associated to the connection when sending the message back to the client.
        Connection connection;
        const CString payload;

        static Message fail(CommandResult::ErrorCode, std::optional<Connection>, std::optional<String> errorMessage = std::nullopt, std::optional<int> commandId = std::nullopt);
        static Message reply(const String& type, unsigned id, Ref<JSON::Value>&& result);
    };

    virtual bool acceptHandshake(HTTPRequestHandler::Request&&) = 0;
    virtual void handleMessage(Message&&, Function<void(Message&&)>&& replyHandler) = 0;
    virtual void clientDisconnected(const Connection&) = 0;
private:
};

class WebSocketServer : public RefCountedAndCanMakeWeakPtr<WebSocketServer> {
public:
    explicit WebSocketServer(WebSocketMessageHandler&, WebDriverService&);
    virtual ~WebSocketServer() = default;

    std::optional<String> listen(const String& host, unsigned port);
    void disconnect();

    WebSocketMessageHandler& messageHandler() { return m_messageHandler; }

    const RefPtr<WebSocketListener>& listener() const { return m_listener; }

    void addStaticConnection(WebSocketMessageHandler::Connection&&);
    bool isStaticConnection(const WebSocketMessageHandler::Connection&);
    void removeStaticConnection(const WebSocketMessageHandler::Connection&);

    void addConnection(WebSocketMessageHandler::Connection&&, const String& sessionId);
    RefPtr<Session> session(const WebSocketMessageHandler::Connection&);
    std::optional<WebSocketMessageHandler::Connection> connection(const String& sessionId);
    void removeConnection(const WebSocketMessageHandler::Connection&);

    RefPtr<WebSocketListener> startListening(const String& sessionId);
    String getResourceName(const String& sessionId);
    String getWebSocketURL(const RefPtr<WebSocketListener>, const String& sessionId);
    String getSessionID(const String& resource);
    void sendMessage(const String& session, const String& message);

    // Non-spec method
    void removeResourceForSession(const String& sessionId);
    void disconnectSession(const String& sessionId);

private:

    WebSocketMessageHandler& m_messageHandler;
    WebDriverService& m_service;
    String m_listenerURL;
    RefPtr<WebSocketListener> m_listener;
    HashMap<WebSocketMessageHandler::Connection, String> m_connectionToSession;

#if USE(SOUP)
    GRefPtr<SoupServer> m_soupServer;
    std::vector<GRefPtr<SoupWebsocketConnection>> m_staticConnections;
#endif
};

}
