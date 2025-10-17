/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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

#include "JSWebExtensionAPIDevToolsInspectedWindow.h"
#include "WebExtensionAPIEvent.h"
#include "WebExtensionAPIObject.h"
#include "WebPageProxyIdentifier.h"

namespace WebKit {

class WebPage;

class WebExtensionAPIDevToolsInspectedWindow : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIDevToolsInspectedWindow, devToolsInspectedWindow, devtools.inspectedWindow);

public:
#if PLATFORM(COCOA)
    void eval(WebPageProxyIdentifier, NSString *expression, NSDictionary *options, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void reload(WebPageProxyIdentifier, NSDictionary *options, NSString **outExceptionString);

    double tabId(WebPage&);
#endif

private:
    RefPtr<WebExtensionAPIEvent> m_onResourceAdded;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)
