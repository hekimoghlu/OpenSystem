/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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

#if ENABLE(WK_WEB_EXTENSIONS)

#include "APIObject.h"
#include "WebExtensionPortChannelIdentifier.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>

OBJC_CLASS NSError;
OBJC_CLASS WKWebExtensionMessagePort;

namespace WebKit {

class WebExtensionContext;

class WebExtensionMessagePort : public API::ObjectImpl<API::Object::Type::WebExtensionMessagePort>, public CanMakeWeakPtr<WebExtensionMessagePort> {
    WTF_MAKE_NONCOPYABLE(WebExtensionMessagePort);

public:
    template<typename... Args>
    static Ref<WebExtensionMessagePort> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionMessagePort(std::forward<Args>(args)...));
    }

    explicit WebExtensionMessagePort(WebExtensionContext&, String applicationIdentifier, WebExtensionPortChannelIdentifier);

    ~WebExtensionMessagePort();

    enum class ErrorType : uint8_t {
        Unknown = 1,
        NotConnected,
        MessageInvalid,
    };

    using Error = std::optional<std::pair<ErrorType, std::optional<String>>>;

    bool operator==(const WebExtensionMessagePort&) const;

    const String& applicationIdentifier() const { return m_applicationIdentifier; }
    WebExtensionPortChannelIdentifier channelIdentifier() const { return m_channelIdentifier; }
    WebExtensionContext* extensionContext() const;

    void disconnect(Error = std::nullopt);
    void reportDisconnection(Error);
    bool isDisconnected() const;

#if PLATFORM(COCOA)
    void sendMessage(id message, CompletionHandler<void(Error)>&& = { });
    void receiveMessage(id message, Error);
#endif

#ifdef __OBJC__
    WKWebExtensionMessagePort *wrapper() const { return (WKWebExtensionMessagePort *)API::ObjectImpl<API::Object::Type::WebExtensionMessagePort>::wrapper(); }
#endif

private:
    void remove();

    WeakPtr<WebExtensionContext> m_extensionContext;
    String m_applicationIdentifier;
    WebExtensionPortChannelIdentifier m_channelIdentifier;
};

NSError *toAPI(WebExtensionMessagePort::Error);
WebExtensionMessagePort::Error toWebExtensionMessagePortError(NSError *);

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
