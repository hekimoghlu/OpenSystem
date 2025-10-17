/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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

#if ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)

#include "JSWebExtensionAPISidePanel.h"
#include "WebExtensionAPIObject.h"

namespace WebKit {

class WebExtensionAPISidePanel : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPISidePanel, sidePanel, sidePanel);

public:
#if PLATFORM(COCOA)
    void getOptions(NSDictionary *options, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void setOptions(NSDictionary *options, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void getPanelBehavior(Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void setPanelBehavior(NSDictionary *behavior, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void open(NSDictionary *options, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
#endif // PLATFORM(COCOA)
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)
