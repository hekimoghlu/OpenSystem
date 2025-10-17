/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
#include "WebExtensionContextProxy.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#include "JSWebExtensionAPINamespace.h"
#include "JSWebExtensionAPIWebPageNamespace.h"
#include "JSWebExtensionWrapper.h"
#include "WebExtensionAPINamespace.h"
#include "WebExtensionAPIWebPageNamespace.h"
#include "WebExtensionControllerProxy.h"
#include "WebFrame.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebExtensionContextProxy);

WebExtensionControllerProxy* WebExtensionContextProxy::extensionControllerProxy() const
{
    return m_extensionControllerProxy.get();
}

void WebExtensionContextProxy::addFrameWithExtensionContent(WebFrame& frame)
{
    m_extensionContentFrames.add(frame);
}

std::optional<WebExtensionTabIdentifier> WebExtensionContextProxy::tabIdentifier(WebPage& page) const
{
    if (m_popupPageMap.contains(page))
        return std::get<std::optional<WebExtensionTabIdentifier>>(m_popupPageMap.get(page));

    if (m_tabPageMap.contains(page))
        return std::get<std::optional<WebExtensionTabIdentifier>>(m_tabPageMap.get(page));

#if ENABLE(INSPECTOR_EXTENSIONS)
    if (m_inspectorPageMap.contains(page))
        return std::get<std::optional<WebExtensionTabIdentifier>>(m_inspectorPageMap.get(page));

    if (m_inspectorBackgroundPageMap.contains(page))
        return std::get<std::optional<WebExtensionTabIdentifier>>(m_inspectorBackgroundPageMap.get(page));
#endif

    return std::nullopt;
}

bool WebExtensionContextProxy::inTestingMode() const
{
    return m_extensionControllerProxy && m_extensionControllerProxy->inTestingMode();
}

RefPtr<WebPage> WebExtensionContextProxy::backgroundPage() const
{
    return m_backgroundPage.get();
}

void WebExtensionContextProxy::setBackgroundPage(WebPage& page)
{
    m_backgroundPage = page;
}

#if ENABLE(INSPECTOR_EXTENSIONS)
void WebExtensionContextProxy::addInspectorPage(WebPage& page, std::optional<WebExtensionTabIdentifier> tabIdentifier, std::optional<WebExtensionWindowIdentifier> windowIdentifier)
{
    m_inspectorPageMap.set(page, TabWindowIdentifierPair { tabIdentifier, windowIdentifier });
}

void WebExtensionContextProxy::addInspectorPageIdentifier(WebCore::PageIdentifier pageIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, std::optional<WebExtensionWindowIdentifier> windowIdentifier)
{
    if (RefPtr page = WebProcess::singleton().webPage(pageIdentifier))
        addInspectorPage(*page, tabIdentifier, windowIdentifier);
}

void WebExtensionContextProxy::addInspectorBackgroundPageIdentifier(WebCore::PageIdentifier pageIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, std::optional<WebExtensionWindowIdentifier> windowIdentifier)
{
    if (RefPtr page = WebProcess::singleton().webPage(pageIdentifier))
        addInspectorBackgroundPage(*page, tabIdentifier, windowIdentifier);
}

void WebExtensionContextProxy::addInspectorBackgroundPage(WebPage& page, std::optional<WebExtensionTabIdentifier> tabIdentifier, std::optional<WebExtensionWindowIdentifier> windowIdentifier)
{
    m_inspectorBackgroundPageMap.set(page, TabWindowIdentifierPair { tabIdentifier, windowIdentifier });
}

bool WebExtensionContextProxy::isInspectorBackgroundPage(WebPage& page) const
{
    return m_inspectorBackgroundPageMap.contains(page);
}
#endif // ENABLE(INSPECTOR_EXTENSIONS)

Vector<Ref<WebPage>> WebExtensionContextProxy::popupPages(std::optional<WebExtensionTabIdentifier> tabIdentifier, std::optional<WebExtensionWindowIdentifier> windowIdentifier) const
{
    Vector<Ref<WebPage>> result;

    for (auto entry : m_popupPageMap) {
        if (tabIdentifier && entry.value.first && entry.value.first.value() != tabIdentifier.value())
            continue;

        if (windowIdentifier && entry.value.second && entry.value.second.value() != windowIdentifier.value())
            continue;

        result.append(Ref { entry.key });
    }

    return result;
}

void WebExtensionContextProxy::addPopupPage(WebPage& page, std::optional<WebExtensionTabIdentifier> tabIdentifier, std::optional<WebExtensionWindowIdentifier> windowIdentifier)
{
    m_popupPageMap.set(page, TabWindowIdentifierPair { tabIdentifier, windowIdentifier });
}

Vector<Ref<WebPage>> WebExtensionContextProxy::tabPages(std::optional<WebExtensionTabIdentifier> tabIdentifier, std::optional<WebExtensionWindowIdentifier> windowIdentifier) const
{
    Vector<Ref<WebPage>> result;

    for (auto entry : m_tabPageMap) {
        if (tabIdentifier && entry.value.first && entry.value.first.value() != tabIdentifier.value())
            continue;

        if (windowIdentifier && entry.value.second && entry.value.second.value() != windowIdentifier.value())
            continue;

        result.append(Ref { entry.key });
    }

    return result;
}

void WebExtensionContextProxy::addTabPage(WebPage& page, std::optional<WebExtensionTabIdentifier> tabIdentifier, std::optional<WebExtensionWindowIdentifier> windowIdentifier)
{
    m_tabPageMap.set(page, TabWindowIdentifierPair { tabIdentifier, windowIdentifier });
}

void WebExtensionContextProxy::setBackgroundPageIdentifier(WebCore::PageIdentifier pageIdentifier)
{
    if (RefPtr page = WebProcess::singleton().webPage(pageIdentifier))
        setBackgroundPage(*page);
}

void WebExtensionContextProxy::addPopupPageIdentifier(WebCore::PageIdentifier pageIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, std::optional<WebExtensionWindowIdentifier> windowIdentifier)
{
    if (RefPtr page = WebProcess::singleton().webPage(pageIdentifier))
        addPopupPage(*page, tabIdentifier, windowIdentifier);
}

void WebExtensionContextProxy::addTabPageIdentifier(WebCore::PageIdentifier pageIdentifier, WebExtensionTabIdentifier tabIdentifier, std::optional<WebExtensionWindowIdentifier> windowIdentifier)
{
    if (RefPtr page = WebProcess::singleton().webPage(pageIdentifier))
        addTabPage(*page, tabIdentifier, windowIdentifier);
}

void WebExtensionContextProxy::setStorageAccessLevel(bool allowedInContentScripts)
{
    m_isSessionStorageAllowedInContentScripts = allowedInContentScripts;
}

void WebExtensionContextProxy::enumerateFramesAndNamespaceObjects(const Function<void(WebFrame&, WebExtensionAPINamespace&)>& function, Ref<DOMWrapperWorld>&& world)
{
    m_extensionContentFrames.forEach([&](auto& frame) {
        RefPtr page = frame.page() ? frame.page()->corePage() : nullptr;
        if (!page)
            return;

        auto context = page->isServiceWorkerPage() ? frame.jsContextForServiceWorkerWorld(world) : frame.jsContextForWorld(world);
        auto globalObject = JSContextGetGlobalObject(context);

        RefPtr<WebExtensionAPINamespace> namespaceObjectImpl;
        auto browserNamespaceObject = JSObjectGetProperty(context, globalObject, toJSString("browser").get(), nullptr);
        if (browserNamespaceObject && JSValueIsObject(context, browserNamespaceObject))
            namespaceObjectImpl = toWebExtensionAPINamespace(context, browserNamespaceObject);

        if (!namespaceObjectImpl) {
            auto chromeNamespaceObject = JSObjectGetProperty(context, globalObject, toJSString("chrome").get(), nullptr);
            if (chromeNamespaceObject && JSValueIsObject(context, chromeNamespaceObject))
                namespaceObjectImpl = toWebExtensionAPINamespace(context, chromeNamespaceObject);
        }

        if (!namespaceObjectImpl)
            return;

        function(frame, *namespaceObjectImpl);
    });
}

void WebExtensionContextProxy::enumerateFramesAndWebPageNamespaceObjects(const Function<void(WebFrame&, WebExtensionAPIWebPageNamespace&)>& function)
{
    m_extensionContentFrames.forEach([&](auto& frame) {
        auto context = frame.jsContextForWorld(mainWorldSingleton());
        auto globalObject = JSContextGetGlobalObject(context);

        RefPtr<WebExtensionAPIWebPageNamespace> namespaceObjectImpl;
        auto browserNamespaceObject = JSObjectGetProperty(context, globalObject, toJSString("browser").get(), nullptr);
        if (browserNamespaceObject && JSValueIsObject(context, browserNamespaceObject))
            namespaceObjectImpl = toWebExtensionAPIWebPageNamespace(context, browserNamespaceObject);

        if (!namespaceObjectImpl)
            return;

        function(frame, *namespaceObjectImpl);
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
