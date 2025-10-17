/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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

#include "CookieRequestHeaderFieldProxy.h"
#include <wtf/URL.h>
#include "ResourceResponse.h"
#include "WebSocketExtensionDispatcher.h"
#include "WebSocketExtensionProcessor.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ResourceRequest;

class WebSocketHandshake {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(WebSocketHandshake, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(WebSocketHandshake);
public:
    enum Mode {
        Incomplete, Normal, Failed, Connected
    };
    WEBCORE_EXPORT WebSocketHandshake(const URL&, const String& protocol, const String& userAgent, const String& clientOrigin, bool allowCookies, bool isAppInitiated);
    WEBCORE_EXPORT ~WebSocketHandshake();

    WEBCORE_EXPORT const URL& url() const;
    void setURL(const URL&);
    WEBCORE_EXPORT URL httpURLForAuthenticationAndCookies() const;
    const String host() const;

    const String& clientProtocol() const;
    void setClientProtocol(const String&);

    bool secure() const;

    String clientLocation() const;

    WEBCORE_EXPORT CString clientHandshakeMessage() const;
    WEBCORE_EXPORT ResourceRequest clientHandshakeRequest(const Function<String(const URL&)>& cookieRequestHeaderFieldValue) const;

    WEBCORE_EXPORT void reset();

    WEBCORE_EXPORT int readServerHandshake(std::span<const uint8_t> header);
    WEBCORE_EXPORT Mode mode() const;
    WEBCORE_EXPORT String failureReason() const; // Returns a string indicating the reason of failure if mode() == Failed.

    WEBCORE_EXPORT String serverWebSocketProtocol() const;
    WEBCORE_EXPORT String serverSetCookie() const;
    String serverUpgrade() const;
    String serverConnection() const;
    String serverWebSocketAccept() const;
    WEBCORE_EXPORT String acceptedExtensions() const;

    WEBCORE_EXPORT const ResourceResponse& serverHandshakeResponse() const;

    WEBCORE_EXPORT void addExtensionProcessor(std::unique_ptr<WebSocketExtensionProcessor>);

    static String getExpectedWebSocketAccept(const String& secWebSocketKey);

private:

    int readStatusLine(std::span<const uint8_t> header, int& statusCode, String& statusText);

    // Reads all headers except for the two predefined ones.
    std::span<const uint8_t> readHTTPHeaders(std::span<const uint8_t>);
    void processHeaders();
    bool checkResponseHeaders();

    URL m_url;
    String m_clientProtocol;
    bool m_secure;

    Mode m_mode;
    String m_userAgent;
    String m_clientOrigin;
    bool m_allowCookies;
    bool m_isAppInitiated;

    ResourceResponse m_serverHandshakeResponse;

    String m_failureReason;

    String m_secWebSocketKey;
    String m_expectedAccept;

    WebSocketExtensionDispatcher m_extensionDispatcher;
};

} // namespace WebCore
