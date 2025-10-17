/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#include "WebExtensionControllerProxy.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#include "WebExtensionContextProxy.h"
#include "WebExtensionControllerMessages.h"
#include "WebExtensionControllerProxyMessages.h"
#include "WebFrame.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

static HashMap<WebExtensionControllerIdentifier, WeakPtr<WebExtensionControllerProxy>>& webExtensionControllerProxies()
{
    static MainThreadNeverDestroyed<HashMap<WebExtensionControllerIdentifier, WeakPtr<WebExtensionControllerProxy>>> controllers;
    return controllers;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebExtensionControllerProxy);

RefPtr<WebExtensionControllerProxy> WebExtensionControllerProxy::get(WebExtensionControllerIdentifier identifier)
{
    return webExtensionControllerProxies().get(identifier).get();
}

Ref<WebExtensionControllerProxy> WebExtensionControllerProxy::getOrCreate(const WebExtensionControllerParameters& parameters, WebPage* newPage)
{
    auto updateProperties = [&](WebExtensionControllerProxy& controller) {
        WebExtensionContextProxySet contexts;
        WebExtensionContextProxyBaseURLMap baseURLMap;

        for (auto& contextParameters : parameters.contextParameters) {
            Ref context = WebExtensionContextProxy::getOrCreate(contextParameters, controller, newPage);
            baseURLMap.add(contextParameters.baseURL.protocolHostAndPort(), context);
            contexts.add(context);
        }

        controller.m_testingMode = parameters.testingMode;
        controller.m_extensionContexts = WTFMove(contexts);
        controller.m_extensionContextBaseURLMap = WTFMove(baseURLMap);
    };

    if (RefPtr controller = get(parameters.identifier)) {
        updateProperties(*controller);
        return *controller;
    }

    Ref result = adoptRef(*new WebExtensionControllerProxy(parameters));
    updateProperties(result);
    return result;
}

WebExtensionControllerProxy::WebExtensionControllerProxy(const WebExtensionControllerParameters& parameters)
    : m_identifier(parameters.identifier)
{
    ASSERT(!get(m_identifier));
    webExtensionControllerProxies().add(m_identifier, *this);

    WebProcess::singleton().addMessageReceiver(Messages::WebExtensionControllerProxy::messageReceiverName(), m_identifier, *this);
}

WebExtensionControllerProxy::~WebExtensionControllerProxy()
{
    webExtensionControllerProxies().remove(m_identifier);
    WebProcess::singleton().removeMessageReceiver(*this);
}

void WebExtensionControllerProxy::load(const WebExtensionContextParameters& contextParameters)
{
    auto context = WebExtensionContextProxy::getOrCreate(contextParameters, *this);
    m_extensionContextBaseURLMap.add(contextParameters.baseURL.protocolHostAndPort(), context);
    m_extensionContexts.add(context);
}

void WebExtensionControllerProxy::unload(WebExtensionContextIdentifier contextIdentifier)
{
    m_extensionContextBaseURLMap.removeIf([&](auto& entry) {
        return entry.value->identifier() == contextIdentifier;
    });

    m_extensionContexts.removeIf([&](auto& entry) {
        return entry->identifier() == contextIdentifier;
    });
}

RefPtr<WebExtensionContextProxy> WebExtensionControllerProxy::extensionContext(const String& uniqueIdentifier) const
{
    for (auto& extensionContext : m_extensionContexts) {
        if (extensionContext->uniqueIdentifier() == uniqueIdentifier)
            return extensionContext.ptr();
    }

    return nullptr;
}

RefPtr<WebExtensionContextProxy> WebExtensionControllerProxy::extensionContext(const URL& url) const
{
    return m_extensionContextBaseURLMap.get(url.protocolHostAndPort());
}

RefPtr<WebExtensionContextProxy> WebExtensionControllerProxy::extensionContext(WebFrame& frame, DOMWrapperWorld& world) const
{
    if (!world.isNormal()) {
        auto prefix = "WebExtension-"_s;
        if (!world.name().startsWith(prefix))
            return nullptr;

        auto prefixLength = prefix.length();
        auto uniqueIdentifier = world.name().substring(prefixLength);
        return extensionContext(uniqueIdentifier);
    }

    return extensionContext(frame.url());
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
