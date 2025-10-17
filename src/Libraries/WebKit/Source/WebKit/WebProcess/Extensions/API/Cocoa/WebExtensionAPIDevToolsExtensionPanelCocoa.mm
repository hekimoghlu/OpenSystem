/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
#import "WebExtensionAPIDevToolsExtensionPanel.h"

#if ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)

#import "CocoaHelpers.h"
#import "JSWebExtensionWrapper.h"
#import "MessageSenderInlines.h"
#import "WebExtensionAPINamespace.h"
#import "WebFrame.h"
#import "WebProcess.h"

namespace WebKit {

WebExtensionAPIEvent& WebExtensionAPIDevToolsExtensionPanel::onShown()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/panels/ExtensionPanel

    if (!m_onShown)
        m_onShown = WebExtensionAPIEvent::create(*this, WebExtensionEventListenerType::DevToolsExtensionPanelOnShown);

    return *m_onShown;
}

WebExtensionAPIEvent& WebExtensionAPIDevToolsExtensionPanel::onHidden()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/panels/ExtensionPanel

    if (!m_onHidden)
        m_onHidden = WebExtensionAPIEvent::create(*this, WebExtensionEventListenerType::DevToolsExtensionPanelOnHidden);

    return *m_onHidden;
}

void WebExtensionContextProxy::dispatchDevToolsExtensionPanelShownEvent(Inspector::ExtensionTabID identifier, WebCore::FrameIdentifier frameIdentifier)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/panels/ExtensionPanel

    RefPtr frame = WebProcess::singleton().webFrame(frameIdentifier);
    if (!frame)
        return;

    enumerateNamespaceObjects([&](auto& namespaceObject) {
        RefPtr extensionPanel = namespaceObject.devtools().panels().extensionPanel(identifier);
        if (!extensionPanel)
            return;

        for (auto& listener : extensionPanel->onShown().listeners()) {
            auto globalContext = listener->globalContext();
            auto *windowObject = toWindowObject(globalContext, *frame) ?: NSNull.null;
            listener->call(windowObject);
        }
    });
}

void WebExtensionContextProxy::dispatchDevToolsExtensionPanelHiddenEvent(Inspector::ExtensionTabID identifier)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/panels/ExtensionPanel

    enumerateNamespaceObjects([&](auto& namespaceObject) {
        RefPtr extensionPanel = namespaceObject.devtools().panels().extensionPanel(identifier);
        if (!extensionPanel)
            return;

        extensionPanel->onHidden().invokeListeners();
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)
