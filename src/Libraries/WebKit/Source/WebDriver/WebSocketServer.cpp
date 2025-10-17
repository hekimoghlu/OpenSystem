/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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
#include "config.h"
#include "WebSocketServer.h"

#include "CommandResult.h"
#include "Session.h"
#include "WebDriverService.h"
#include <algorithm>
#include <optional>
#include <wtf/JSONValues.h>
#include <wtf/UUID.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebDriver {

WebSocketServer::WebSocketServer(WebSocketMessageHandler& messageHandler, WebDriverService& service)
    : m_messageHandler(messageHandler)
    , m_service(service)
{
}

void WebSocketServer::addStaticConnection(WebSocketMessageHandler::Connection&& connection)
{
    m_staticConnections.push_back(WTFMove(connection));
}

void WebSocketServer::addConnection(WebSocketMessageHandler::Connection&& connection, const String& sessionId)
{
    m_connectionToSession.add(WTFMove(connection), sessionId);
}

bool WebSocketServer::isStaticConnection(const WebSocketMessageHandler::Connection& connection)
{
    return std::count(m_staticConnections.begin(), m_staticConnections.end(), connection);
}

void WebSocketServer::removeStaticConnection(const WebSocketMessageHandler::Connection& connection)
{
    m_staticConnections.erase(std::find(m_staticConnections.begin(), m_staticConnections.end(), connection));
}

void WebSocketServer::removeConnection(const WebSocketMessageHandler::Connection& connection)
{
    auto it = m_connectionToSession.find(connection);
    if (it != m_connectionToSession.end())
        m_connectionToSession.remove(it);
}

RefPtr<Session> WebSocketServer::session(const WebSocketMessageHandler::Connection& connection)
{
    String sessionId;

    for (const auto& pair : m_connectionToSession) {
        if (pair.key == connection) {
            sessionId = pair.value;
            break;
        }
    }

    if (sessionId.isNull())
        return { };

    const auto& existingSession = m_service.session();
    if (!existingSession || (existingSession->id() != sessionId))
        return { };

    return existingSession;
}

std::optional<WebSocketMessageHandler::Connection> WebSocketServer::connection(const String& sessionId)
{
    for (auto& pair : m_connectionToSession) {
        if (pair.value == sessionId)
            return pair.key;
    }

    return { };
}

String WebSocketServer::getSessionID(const String& resource)
{
    // https://w3c.github.io/webdriver-bidi/#get-a-session-id-for-a-websocket-resource

    constexpr auto sessionPrefix = "/session/"_s;
    if (!resource.startsWith(sessionPrefix))
        return nullString();

    auto sessionId = resource.substring(sessionPrefix.length());
    auto uuid = WTF::UUID::parse(sessionId);
    if (!uuid || !uuid->isValid())
        return nullString();

    return sessionId;
}

RefPtr<WebSocketListener> WebSocketServer::startListening(const String& sessionId)
{
    // https://w3c.github.io/webdriver-bidi/#start-listening-for-a-websocket-connection
    // FIXME Add more listeners when start supporting multiple sessions
    auto resourceName = getResourceName(sessionId);
    m_listener->resources.push_back(resourceName);
    return m_listener;
}

String WebSocketServer::getResourceName(const String& sessionId)
{
    // https://w3c.github.io/webdriver-bidi/#construct-a-websocket-resource-name
    if (sessionId.isNull())
        return "/session"_s;

    return makeString("/session/"_s, sessionId);
}

String WebSocketServer::getWebSocketURL(const RefPtr<WebSocketListener> listener, const String& sessionId)
{
    // https://w3c.github.io/webdriver-bidi/#construct-a-websocket-url
    // FIXME: Support secure flag
    // https://bugs.webkit.org/show_bug.cgi?id=280680

    auto resourceName = getResourceName(sessionId);

    auto host = listener->host;
    if (host == "all"_s)
        host = "localhost"_s;

    return makeString("ws://"_s, host, ":"_s, String::number(listener->port), resourceName);
}

void WebSocketServer::removeResourceForSession(const String& sessionId)
{
    auto resourceName = getResourceName(sessionId);
    m_listener->resources.erase(std::remove(m_listener->resources.begin(), m_listener->resources.end(), resourceName), m_listener->resources.end());
}

WebSocketMessageHandler::Message WebSocketMessageHandler::Message::fail(CommandResult::ErrorCode errorCode, std::optional<Connection> connection, std::optional<String> errorMessage, std::optional<int> commandId)
{
    auto reply = JSON::Object::create();

    if (commandId)
        reply->setInteger("id"_s, *commandId);

    if (errorMessage)
        reply->setString("message"_s, *errorMessage);

    reply->setInteger("error"_s, CommandResult::errorCodeToHTTPStatusCode(errorCode));

    return { (connection ? (*connection) : nullptr), reply->toJSONString().utf8() };
}

WebSocketMessageHandler::Message WebSocketMessageHandler::Message::reply(const String& type, unsigned id, Ref<JSON::Value>&& result)
{
    auto reply = JSON::Object::create();
    reply->setString("type"_s, type);
    reply->setInteger("id"_s, id);

    if (auto resultObject = result->asObject())
        reply->setObject("result"_s, resultObject.releaseNonNull());
    else
        reply->setObject("result"_s, JSON::Object::create());

    // Connection will be set when sending the message back to the client, from the incoming message.
    return { nullptr, reply->toJSONString().utf8() };
}

std::optional<Command> Command::fromData(const char* data, size_t dataLength)
{
    auto messageValue = JSON::Value::parseJSON(String::fromUTF8(std::span<const char>(data, dataLength)));
    if (!messageValue)
        return std::nullopt;

    auto messageObject = messageValue->asObject();
    if (!messageObject)
        return std::nullopt;

    auto id = messageObject->getString("id"_s);
    if (!id)
        return std::nullopt;

    auto method = messageObject->getString("method"_s);
    if (!method)
        return std::nullopt;

    auto params = messageObject->getObject("params"_s);
    if (!params)
        return std::nullopt;

    Command command {
        .id = id,
        .method = method,
        .parameters = params
    };

    return command;
}


} // namespace WebDriver
