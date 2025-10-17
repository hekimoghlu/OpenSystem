/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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

#include "JSWebExtensionAPIAction.h"
#include "WebExtensionAPIEvent.h"
#include "WebExtensionAPIObject.h"
#include "WebPageProxyIdentifier.h"

OBJC_CLASS NSDictionary;
OBJC_CLASS NSString;

namespace WebKit {

class WebExtensionAPIAction : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIAction, action, action);

public:
#if PLATFORM(COCOA)
    void getTitle(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void setTitle(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void getBadgeText(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void setBadgeText(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void getBadgeBackgroundColor(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void setBadgeBackgroundColor(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void enable(double tabIdentifier, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void disable(double tabIdentifier, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void isEnabled(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void setIcon(WebFrame&, NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void getPopup(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void setPopup(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void openPopup(WebPageProxyIdentifier, NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    WebExtensionAPIEvent& onClicked();

private:
    friend class WebExtensionAPIMenus;

    static bool isValidDimensionKey(NSString *);
    static NSString *parseIconPath(NSString *path, const URL& baseURL);
    static NSMutableDictionary *parseIconPathsDictionary(NSDictionary *, const URL& baseURL, bool forVariants, NSString *inputKey, NSString **outExceptionString);
    static NSMutableDictionary *parseIconImageDataDictionary(NSDictionary *, bool forVariants, NSString *inputKey, NSString **outExceptionString);

#if ENABLE(WK_WEB_EXTENSIONS_ICON_VARIANTS)
    static NSArray *parseIconVariants(NSArray *, const URL& baseURL, NSString *inputKey, NSString **outExceptionString);
#endif

    static bool parseActionDetails(NSDictionary *, std::optional<WebExtensionWindowIdentifier>&, std::optional<WebExtensionTabIdentifier>&, NSString **outExceptionString);

    RefPtr<WebExtensionAPIEvent> m_onClicked;
#endif
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
