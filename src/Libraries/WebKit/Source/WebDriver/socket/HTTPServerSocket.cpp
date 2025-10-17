/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 29, 2024.
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
#include "HTTPServer.h"
#include <wtf/text/StringBuilder.h>

namespace WebDriver {

bool HTTPServer::listen(const std::optional<String>& host, unsigned port)
{
    auto& endpoint = RemoteInspectorSocketEndpoint::singleton();

    if (auto id = endpoint.listenInet(host ? host.value().utf8().data() : "", port, *this)) {
        m_server = id;
        return true;
    }
    return false;
}

void HTTPServer::disconnect()
{
    auto& endpoint = RemoteInspectorSocketEndpoint::singleton();
    endpoint.disconnect(m_server.value());
}

std::optional<ConnectionID> HTTPServer::doAccept(RemoteInspectorSocketEndpoint& endpoint, PlatformSocketType socket)
{
    if (auto id = endpoint.createClient(socket, m_requestHandler)) {
        m_requestHandler.connect(id.value());
        return id;
    }

    return std::nullopt;
}

void HTTPServer::didChangeStatus(RemoteInspectorSocketEndpoint&, ConnectionID, RemoteInspectorSocketEndpoint::Listener::Status status)
{
    if (status == Status::Closed)
        m_server = std::nullopt;
}

void HTTPRequestHandler::connect(ConnectionID id)
{
    m_client = id;
    reset();
}

void HTTPRequestHandler::reset()
{
    m_parser = { };
}

void HTTPRequestHandler::didReceive(RemoteInspectorSocketEndpoint&, ConnectionID, Vector<uint8_t>&& data)
{
    switch (m_parser.parse(WTFMove(data))) {
    case HTTPParser::Phase::Complete: {
        auto message = m_parser.pullMessage();
        HTTPRequestHandler::Request request {
            message.method,
            message.path,
            reinterpret_cast<const char*>(message.requestBody.data()),
            static_cast<size_t>(message.requestBody.size())
        };

        handleRequest(WTFMove(request), [this](HTTPRequestHandler::Response&& response) {
            sendResponse(WTFMove(response));
        });
        break;
    }
    case HTTPParser::Phase::Error: {
        HTTPRequestHandler::Response response {
            400,
            "text/html; charset=utf-8",
            "<h1>Bad client</h1> Invalid HTML format"_s,
        };
        sendResponse(WTFMove(response));
        return;
    }
    default:
        return;
    }
}

void HTTPRequestHandler::sendResponse(HTTPRequestHandler::Response&& response)
{
    auto& endpoint = RemoteInspectorSocketEndpoint::singleton();
    endpoint.send(m_client.value(), packHTTPMessage(WTFMove(response)).utf8().span());
    reset();
}

String HTTPRequestHandler::packHTTPMessage(HTTPRequestHandler::Response&& response) const
{
    StringBuilder builder;
    auto EOL = "\r\n"_s;

    builder.append("HTTP/1.0 "_s, response.statusCode, ' ', response.statusCode == 200 ? "OK"_s : "ERROR"_s, EOL);

    if (!response.data.isNull()) {
        builder.append("Content-Type: "_s, response.contentType, EOL,
            "Content-Length: "_s, response.data.length(), EOL,
            "Cache-Control: no-cache"_s, EOL);
    }

    builder.append(EOL);

    if (!response.data.isNull())
        builder.append(String::fromUTF8(response.data.span()));

    return builder.toString();
}

void HTTPRequestHandler::didClose(RemoteInspectorSocketEndpoint&, ConnectionID)
{
    m_client = std::nullopt;
    reset();
}

} // namespace WebDriver
