/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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

#include "JSWebExtensionAPIPort.h"
#include "WebExtensionAPIEvent.h"
#include "WebExtensionAPIObject.h"
#include "WebExtensionMessageSenderParameters.h"
#include "WebExtensionPortChannelIdentifier.h"
#include "WebPageProxyIdentifier.h"
#include <wtf/WeakPtr.h>

OBJC_CLASS NSDictionary;
OBJC_CLASS NSObject;
OBJC_CLASS NSString;

namespace WebKit {

class WebExtensionAPIPort : public WebExtensionAPIObject, public JSWebExtensionWrappable, public CanMakeWeakPtr<WebExtensionAPIPort> {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIPort, port, port);

public:
#if PLATFORM(COCOA)
    using PortSet = HashSet<Ref<WebExtensionAPIPort>>;

    static PortSet get(WebExtensionPortChannelIdentifier);

    WebExtensionContentWorldType targetContentWorldType() const { return m_targetContentWorldType; }
    WebExtensionPortChannelIdentifier channelIdentifier() const { return *m_channelIdentifier; }
    std::optional<WebPageProxyIdentifier> owningPageProxyIdentifier() const { return m_owningPageProxyIdentifier; }
    const std::optional<WebExtensionMessageSenderParameters>& senderParameters() const { return m_senderParameters; }

    void postMessage(WebFrame&, NSString *, NSString **outExceptionString);
    void disconnect();

    bool isDisconnected() const { return m_disconnected; }
    bool isQuarantined() const { return !m_channelIdentifier; }

    NSString *name();
    NSDictionary *sender();

    JSValue *error();
    void setError(JSValue *);

    WebExtensionAPIEvent& onMessage();
    WebExtensionAPIEvent& onDisconnect();

    virtual ~WebExtensionAPIPort()
    {
        // Don't fire the disconnect event, since this port is being finalized
        // and we can't call into JavaScript during garbage collection.
        m_disconnected = true;

        remove();
    }

private:
    friend class WebExtensionContextProxy;

    explicit WebExtensionAPIPort(const WebExtensionAPIObject& parentObject, const String& name)
        : WebExtensionAPIObject(parentObject)
        , m_targetContentWorldType(WebExtensionContentWorldType::Main)
        , m_name(name)
    {
        ASSERT(isQuarantined());
    }

    explicit WebExtensionAPIPort(const WebExtensionAPIObject& parentObject, WebPageProxyIdentifier owningPageProxyIdentifier, WebExtensionContentWorldType targetContentWorldType, const String& name)
        : WebExtensionAPIObject(parentObject)
        , m_targetContentWorldType(targetContentWorldType)
        , m_owningPageProxyIdentifier(owningPageProxyIdentifier)
        , m_channelIdentifier(WebExtensionPortChannelIdentifier::generate())
        , m_name(name)
    {
        add();
    }

    explicit WebExtensionAPIPort(WebExtensionContentWorldType contentWorldType, WebExtensionAPIRuntimeBase& runtime, WebExtensionContextProxy& context, WebPageProxyIdentifier owningPageProxyIdentifier, WebExtensionContentWorldType targetContentWorldType, const String& name)
        : WebExtensionAPIObject(contentWorldType, runtime, context)
        , m_targetContentWorldType(targetContentWorldType)
        , m_owningPageProxyIdentifier(owningPageProxyIdentifier)
        , m_channelIdentifier(WebExtensionPortChannelIdentifier::generate())
        , m_name(name)
    {
        add();
    }

    explicit WebExtensionAPIPort(const WebExtensionAPIObject& parentObject, WebPageProxyIdentifier owningPageProxyIdentifier, WebExtensionContentWorldType targetContentWorldType, const String& name, const WebExtensionMessageSenderParameters& senderParameters)
        : WebExtensionAPIObject(parentObject)
        , m_targetContentWorldType(targetContentWorldType)
        , m_owningPageProxyIdentifier(owningPageProxyIdentifier)
        , m_channelIdentifier(WebExtensionPortChannelIdentifier::generate())
        , m_name(name)
        , m_senderParameters(senderParameters)
    {
        add();
    }

    explicit WebExtensionAPIPort(const WebExtensionAPIObject& parentObject, WebPageProxyIdentifier owningPageProxyIdentifier, WebExtensionContentWorldType targetContentWorldType, WebExtensionPortChannelIdentifier channelIdentifier, const String& name, const WebExtensionMessageSenderParameters& senderParameters)
        : WebExtensionAPIObject(parentObject)
        , m_targetContentWorldType(targetContentWorldType)
        , m_owningPageProxyIdentifier(owningPageProxyIdentifier)
        , m_channelIdentifier(channelIdentifier)
        , m_name(name)
        , m_senderParameters(senderParameters)
    {
        add();
    }

    void add();
    void remove();

    void fireMessageEventIfNeeded(id message);
    void fireDisconnectEventIfNeeded();

    WebExtensionContentWorldType m_targetContentWorldType;
    Markable<WebPageProxyIdentifier> m_owningPageProxyIdentifier;
    Markable<WebExtensionPortChannelIdentifier> m_channelIdentifier;
    bool m_disconnected { false };

    String m_name;
    RetainPtr<JSValue> m_error;
    std::optional<WebExtensionMessageSenderParameters> m_senderParameters;

    RefPtr<WebExtensionAPIEvent> m_onMessage;
    RefPtr<WebExtensionAPIEvent> m_onDisconnect;
#endif
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
