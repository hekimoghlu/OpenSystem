/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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

#if ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)

#include "JSWebExtensionAPIDevTools.h"
#include "WebExtensionAPIDevToolsInspectedWindow.h"
#include "WebExtensionAPIDevToolsNetwork.h"
#include "WebExtensionAPIDevToolsPanels.h"
#include "WebExtensionAPIObject.h"

namespace WebKit {

class WebExtensionAPIDevTools : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIDevTools, devTools, devtools);

public:
#if PLATFORM(COCOA)
    WebExtensionAPIDevToolsInspectedWindow& inspectedWindow();
    WebExtensionAPIDevToolsNetwork& network();
    WebExtensionAPIDevToolsPanels& panels();
#endif

private:
    RefPtr<WebExtensionAPIDevToolsInspectedWindow> m_inspectedWindow;
    RefPtr<WebExtensionAPIDevToolsNetwork> m_network;
    RefPtr<WebExtensionAPIDevToolsPanels> m_panels;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)
