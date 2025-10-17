/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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

#include "Logging.h"
#include <libsoup/soup.h>
#include <wtf/Function.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebDriver {

static bool soupServerListen(SoupServer* server, const std::optional<String>& host, unsigned port, GError** error)
{
    static const auto options = static_cast<SoupServerListenOptions>(0);
    if (!host || host.value() == "local"_s)
        return soup_server_listen_local(server, port, options, error);

    if (host.value() == "all"_s)
        return soup_server_listen_all(server, port, options, error);

    GRefPtr<GSocketAddress> address = adoptGRef(g_inet_socket_address_new_from_string(host.value().utf8().data(), port));
    if (!address) {
        g_set_error(error, G_IO_ERROR, G_IO_ERROR_INVALID_ARGUMENT, "Invalid host IP address '%s'", host.value().utf8().data());
        return false;
    }

    return soup_server_listen(server, address.get(), options, error);
}

bool HTTPServer::listen(const std::optional<String>& host, unsigned port)
{
    m_soupServer = adoptGRef(soup_server_new("server-header", "WebKitWebDriver", nullptr));
    GUniqueOutPtr<GError> error;
    if (!soupServerListen(m_soupServer.get(), host, port, &error.outPtr())) {
        RELEASE_LOG(WebDriverClassic, "Failed to start HTTP server at port %u: %s", port, error->message);
        return false;
    }

#if USE(SOUP2)
    soup_server_add_handler(m_soupServer.get(), nullptr, [](SoupServer* server, SoupMessage* message, const char* path, GHashTable*, SoupClientContext*, gpointer userData) {
        auto* httpServer = static_cast<HTTPServer*>(userData);
        GRefPtr<SoupMessage> protectedMessage = message;
        soup_server_pause_message(server, message);
        httpServer->m_requestHandler.handleRequest({ String::fromUTF8(message->method), String::fromUTF8(path), message->request_body->data, static_cast<size_t>(message->request_body->length) },
            [server, message = WTFMove(protectedMessage)](HTTPRequestHandler::Response&& response) {
                soup_message_set_status(message.get(), response.statusCode);
                if (!response.data.isNull()) {
                    // Â§6.3 Processing Model.
                    // https://w3c.github.io/webdriver/webdriver-spec.html#dfn-send-a-response
                    soup_message_headers_append(message->response_headers, "Content-Type", response.contentType.utf8().data());
                    soup_message_headers_append(message->response_headers, "Cache-Control", "no-cache");
                    soup_message_body_append(message->response_body, SOUP_MEMORY_COPY, response.data.data(), response.data.length());
                }
                soup_server_unpause_message(server, message.get());
        });
    }, this, nullptr);
#else
    soup_server_add_handler(m_soupServer.get(), nullptr, [](SoupServer* server, SoupServerMessage* message, const char* path, GHashTable*, gpointer userData) {
        auto& httpServer = *static_cast<HTTPServer*>(userData);
        GRefPtr<SoupServerMessage> protectedMessage = message;
        soup_server_pause_message(server, message);
        auto* requestBody = soup_server_message_get_request_body(message);

        if (httpServer.m_visibleHost.isEmpty()) {
            const auto* visibleHostAndPort = soup_message_headers_get_one(soup_server_message_get_request_headers(message), "Host");
            httpServer.m_visibleHost = visibleHostAndPort ? String::fromLatin1(visibleHostAndPort).split(':').at(0) : "localhost"_s;
        }

        httpServer.m_requestHandler.handleRequest({ String::fromUTF8(soup_server_message_get_method(message)), String::fromUTF8(path), requestBody->data, static_cast<size_t>(requestBody->length) },
            [server, message = WTFMove(protectedMessage)](HTTPRequestHandler::Response&& response) {
                soup_server_message_set_status(message.get(), response.statusCode, nullptr);
                if (!response.data.isNull()) {
                    // Â§6.3 Processing Model.
                    // https://w3c.github.io/webdriver/webdriver-spec.html#dfn-send-a-response
                    auto* responseHeaders = soup_server_message_get_response_headers(message.get());
                    soup_message_headers_append(responseHeaders, "Content-Type", response.contentType.utf8().data());
                    soup_message_headers_append(responseHeaders, "Cache-Control", "no-cache");
                    auto* responseBody = soup_server_message_get_response_body(message.get());
                    soup_message_body_append(responseBody, SOUP_MEMORY_COPY, response.data.data(), response.data.length());
                }
                soup_server_unpause_message(server, message.get());
        });
    }, this, nullptr);
#endif

    return true;
}

void HTTPServer::disconnect()
{
    soup_server_disconnect(m_soupServer.get());
    m_soupServer = nullptr;
}

} // namespace WebDriver
