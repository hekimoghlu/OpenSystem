/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#import "WebExtensionAPIDevToolsPanels.h"

#if ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)

#import "CocoaHelpers.h"
#import "InspectorExtensionTypes.h"
#import "JSWebExtensionWrapper.h"
#import "MessageSenderInlines.h"
#import "WebExtensionAPIEvent.h"
#import "WebExtensionAPINamespace.h"
#import "WebExtensionContextMessages.h"
#import "WebProcess.h"

namespace WebKit {

RefPtr<WebExtensionAPIDevToolsExtensionPanel> WebExtensionAPIDevToolsPanels::extensionPanel(Inspector::ExtensionTabID identifier) const
{
    return m_extensionPanels.get(identifier);
}

void WebExtensionAPIDevToolsPanels::createPanel(WebPageProxyIdentifier webPageProxyIdentifier, NSString *title, NSString *iconPath, NSString *pagePath, Ref<WebExtensionCallbackHandler>&& callback, NSString **outExceptionString)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/panels/create

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::DevToolsPanelsCreate(webPageProxyIdentifier, title, iconPath, pagePath), [this, protectedThis = Ref { *this }, callback = WTFMove(callback)](Expected<Inspector::ExtensionTabID, WebExtensionError>&& result) mutable {
        if (!result) {
            callback->reportError(result.error());
            return;
        }

        Ref extensionPanel = WebExtensionAPIDevToolsExtensionPanel::create(*this);
        m_extensionPanels.set(result.value(), extensionPanel);

        auto globalContext = callback->globalContext();
        auto *panelValue = toJSValue(globalContext, toJS(globalContext, extensionPanel.ptr()));

        callback->call(panelValue);
    }, extensionContext().identifier());
}

NSString *WebExtensionAPIDevToolsPanels::themeName()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/panels/themeName

    switch (extensionContext().inspectorAppearance()) {
    case Inspector::ExtensionAppearance::Light:
        return @"light";

    case Inspector::ExtensionAppearance::Dark:
        return @"dark";
    }
}

WebExtensionAPIEvent& WebExtensionAPIDevToolsPanels::onThemeChanged()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/panels/onThemeChanged

    if (!m_onThemeChanged)
        m_onThemeChanged = WebExtensionAPIEvent::create(*this, WebExtensionEventListenerType::DevToolsPanelsOnThemeChanged);

    return *m_onThemeChanged;
}

void WebExtensionContextProxy::dispatchDevToolsPanelsThemeChangedEvent(Inspector::ExtensionAppearance appearance)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/panels/onThemeChanged

    setInspectorAppearance(appearance);

    enumerateNamespaceObjects([&](auto& namespaceObject) {
        auto& panels = namespaceObject.devtools().panels();
        panels.onThemeChanged().invokeListenersWithArgument(panels.themeName());
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)
