/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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

#include <libsoup/soup.h>

#if USE(SOUP2)

static inline const char*
soup_message_get_method(SoupMessage* message)
{
    g_return_val_if_fail(SOUP_IS_MESSAGE(message), nullptr);
    return message->method;
}

static inline const char*
soup_server_message_get_method(SoupMessage* message)
{
    return soup_message_get_method(message);
}

static inline SoupStatus
soup_message_get_status(SoupMessage* message)
{
    g_return_val_if_fail(SOUP_IS_MESSAGE(message), SOUP_STATUS_NONE);
    return static_cast<SoupStatus>(message->status_code);
}

static inline void
soup_server_message_set_status(SoupMessage* message, unsigned statusCode, const char* reasonPhrase)
{
    if (reasonPhrase)
        soup_message_set_status_full(message, statusCode, reasonPhrase);
    else
        soup_message_set_status(message, statusCode);
}

static inline const char*
soup_message_get_reason_phrase(SoupMessage* message)
{
    g_return_val_if_fail(SOUP_IS_MESSAGE(message), nullptr);
    return message->reason_phrase;
}

static inline SoupMessageHeaders*
soup_message_get_request_headers(SoupMessage* message)
{
    g_return_val_if_fail(SOUP_IS_MESSAGE(message), nullptr);
    return message->request_headers;
}

static inline SoupMessageHeaders*
soup_server_message_get_request_headers(SoupMessage* message)
{
    return soup_message_get_request_headers(message);
}

static inline SoupMessageHeaders*
soup_message_get_response_headers(SoupMessage* message)
{
    g_return_val_if_fail(SOUP_IS_MESSAGE(message), nullptr);
    return message->response_headers;
}

static inline SoupMessageHeaders*
soup_server_message_get_response_headers(SoupMessage* message)
{
    return soup_message_get_response_headers(message);
}

static inline SoupMessageBody*
soup_server_message_get_response_body(SoupMessage* message)
{
    g_return_val_if_fail(SOUP_IS_MESSAGE(message), nullptr);
    return message->response_body;
}

static inline void
soup_server_message_set_response(SoupMessage* message, const char* contentType, SoupMemoryUse memoryUse, const char* responseBody, gsize length)
{
    return soup_message_set_response(message, contentType, memoryUse, responseBody, length);
}

static inline SoupURI*
soup_server_message_get_uri(SoupMessage* message)
{
    return soup_message_get_uri(message);
}

static inline GTlsCertificate*
soup_message_get_tls_peer_certificate(SoupMessage* message)
{
    g_return_val_if_fail(SOUP_IS_MESSAGE(message), nullptr);
    GTlsCertificate* certificate = nullptr;
    soup_message_get_https_status(message, &certificate, nullptr);
    return certificate;
}

static inline GTlsCertificateFlags
soup_message_get_tls_peer_certificate_errors(SoupMessage* message)
{
    g_return_val_if_fail(SOUP_IS_MESSAGE(message), static_cast<GTlsCertificateFlags>(0));
    GTlsCertificateFlags flags = static_cast<GTlsCertificateFlags>(0);
    soup_message_get_https_status(message, nullptr, &flags);
    return flags;
}

static inline void
soup_session_send_async(SoupSession* session, SoupMessage* message, int, GCancellable* cancellable, GAsyncReadyCallback callback, gpointer userData)
{
    soup_session_send_async(session, message, cancellable, callback, userData);
}

static inline void
soup_session_websocket_connect_async(SoupSession* session, SoupMessage* message, const char* origin, char** protocols, int, GCancellable* cancellable, GAsyncReadyCallback callback, gpointer userData)
{
    soup_session_websocket_connect_async(session, message, origin, protocols, cancellable, callback, userData);
}

static inline void
soup_auth_cancel(SoupAuth*)
{
}

static inline void
soup_session_set_proxy_resolver(SoupSession* session, GProxyResolver* resolver)
{
    g_object_set(session, "proxy-resolver", resolver, nullptr);
}

static inline GProxyResolver*
soup_session_get_proxy_resolver(SoupSession* session)
{
    GRefPtr<GProxyResolver> resolver;
    g_object_get(session, "proxy-resolver", &resolver.outPtr(), nullptr);
    return resolver.get();
}

static inline void
soup_session_set_accept_language(SoupSession* session, const char* acceptLanguage)
{
    g_object_set(session, "accept-language", acceptLanguage, nullptr);
}

static inline void
soup_session_set_tls_database(SoupSession *session, GTlsDatabase *tls_database)
{
    g_object_set(session, "tls-database", tls_database, NULL);
}

#endif // USE(SOUP2)
