/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionMessagePort.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "CocoaHelpers.h"
#import "WKWebExtensionMessagePortInternal.h"
#import "WebExtensionContext.h"
#import <wtf/BlockPtr.h>

namespace WebKit {

NSError *toAPI(WebExtensionMessagePort::Error error)
{
    if (!error)
        return nil;

    WKWebExtensionMessagePortError errorCode;
    NSString *message;

    switch (error.value().first) {
    case WebKit::WebExtensionMessagePort::ErrorType::Unknown:
        errorCode = WKWebExtensionMessagePortErrorUnknown;
        message = (NSString *)error.value().second.value_or("An unknown error occurred."_s);
        break;

    case WebKit::WebExtensionMessagePort::ErrorType::NotConnected:
        errorCode = WKWebExtensionMessagePortErrorNotConnected;
        message = (NSString *)error.value().second.value_or("Message port is not connected and cannot send messages."_s);
        break;

    case WebKit::WebExtensionMessagePort::ErrorType::MessageInvalid:
        errorCode = WKWebExtensionMessagePortErrorMessageInvalid;
        message = (NSString *)error.value().second.value_or("Message is not JSON-serializable."_s);
        break;
    }

    return [NSError errorWithDomain:WKWebExtensionMessagePortErrorDomain code:errorCode userInfo:@{ NSDebugDescriptionErrorKey: message }];
}

WebExtensionMessagePort::Error toWebExtensionMessagePortError(NSError *error)
{
    if (!error)
        return std::nullopt;
    return { { WebExtensionMessagePort::ErrorType::Unknown, error.localizedDescription } };
}

WebExtensionMessagePort::WebExtensionMessagePort(WebExtensionContext& extensionContext, String applicationIdentifier, WebExtensionPortChannelIdentifier channelIdentifier)
    : m_extensionContext(extensionContext)
    , m_applicationIdentifier(applicationIdentifier)
    , m_channelIdentifier(channelIdentifier)
{
}

WebExtensionMessagePort::~WebExtensionMessagePort()
{
    remove();
}

bool WebExtensionMessagePort::operator==(const WebExtensionMessagePort& other) const
{
    return this == &other || (m_extensionContext == other.m_extensionContext && m_applicationIdentifier == other.m_applicationIdentifier && m_channelIdentifier == other.m_channelIdentifier);
}

WebExtensionContext* WebExtensionMessagePort::extensionContext() const
{
    return m_extensionContext.get();
}

bool WebExtensionMessagePort::isDisconnected() const
{
    return !m_extensionContext;
}

void WebExtensionMessagePort::disconnect(Error error)
{
    remove();
}

void WebExtensionMessagePort::reportDisconnection(Error error)
{
    ASSERT(!isDisconnected());

    remove();

    if (auto disconnectHandler = wrapper().disconnectHandler)
        disconnectHandler(toAPI(error));
}

void WebExtensionMessagePort::remove()
{
    if (isDisconnected())
        return;

    Ref protectedThis { *this };

    RefPtr extensionContext = m_extensionContext.get();
    if (!extensionContext)
        return;

    extensionContext->removeNativePort(*this);
    extensionContext->firePortDisconnectEventIfNeeded(WebExtensionContentWorldType::Native, WebExtensionContentWorldType::Main, m_channelIdentifier);
    m_extensionContext = nullptr;
}

void WebExtensionMessagePort::sendMessage(id message, CompletionHandler<void(Error error)>&& completionHandler)
{
    if (isDisconnected()) {
        if (completionHandler)
            completionHandler({ { ErrorType::NotConnected, std::nullopt } });
        return;
    }

    if (message && !isValidJSONObject(message, JSONOptions::FragmentsAllowed)) {
        if (completionHandler)
            completionHandler({ { ErrorType::MessageInvalid, std::nullopt } });
        return;
    }

    RefPtr extensionContext = m_extensionContext.get();
    if (!extensionContext) {
        if (completionHandler)
            completionHandler({ { ErrorType::NotConnected, std::nullopt } });
        return;
    }

    extensionContext->portPostMessage(WebExtensionContentWorldType::Native, WebExtensionContentWorldType::Main, std::nullopt, m_channelIdentifier, encodeJSONString(message, JSONOptions::FragmentsAllowed) );

    if (completionHandler)
        completionHandler(std::nullopt);
}

void WebExtensionMessagePort::receiveMessage(id message, Error error)
{
    ASSERT(!isDisconnected());

    if (auto messageHandler = wrapper().messageHandler)
        messageHandler(message, toAPI(error));
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
