/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
#import "WebExtensionAPIDevTools.h"

#if ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)

#import "CocoaHelpers.h"
#import "JSWebExtensionWrapper.h"
#import "MessageSenderInlines.h"
#import "WebExtensionAPIDevToolsInspectedWindow.h"
#import "WebExtensionAPIDevToolsNetwork.h"
#import "WebExtensionAPIDevToolsPanels.h"

namespace WebKit {

WebExtensionAPIDevToolsInspectedWindow& WebExtensionAPIDevTools::inspectedWindow()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/inspectedWindow

    if (!m_inspectedWindow)
        m_inspectedWindow = WebExtensionAPIDevToolsInspectedWindow::create(*this);

    return *m_inspectedWindow;
}

WebExtensionAPIDevToolsNetwork& WebExtensionAPIDevTools::network()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/network

    if (!m_network)
        m_network = WebExtensionAPIDevToolsNetwork::create(*this);

    return *m_network;
}

WebExtensionAPIDevToolsPanels& WebExtensionAPIDevTools::panels()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/panels

    if (!m_panels)
        m_panels = WebExtensionAPIDevToolsPanels::create(*this);

    return *m_panels;
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)
