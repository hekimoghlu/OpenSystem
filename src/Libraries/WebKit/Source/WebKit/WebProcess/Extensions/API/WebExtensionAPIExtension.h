/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 13, 2024.
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

#include "JSWebExtensionAPIExtension.h"
#include "WebExtensionAPIObject.h"

OBJC_CLASS NSString;
OBJC_CLASS NSURL;

namespace WebKit {

class WebPage;

class WebExtensionAPIExtension : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIExtension, extension, extension);

public:
    enum class ViewType : uint8_t {
        Popup,
        Tab,
    };

#if PLATFORM(COCOA)
    bool isPropertyAllowed(const ASCIILiteral& propertyName, WebPage*);

    bool isInIncognitoContext(WebPage&);
    void isAllowedFileSchemeAccess(Ref<WebExtensionCallbackHandler>&&);
    void isAllowedIncognitoAccess(Ref<WebExtensionCallbackHandler>&&);

    NSURL *getURL(NSString *resourcePath, NSString **outExceptionString);
    JSValue *getBackgroundPage(JSContextRef);
    NSArray *getViews(JSContextRef, NSDictionary *filter, NSString **outExceptionString);
#endif

private:
    static bool parseViewFilters(NSDictionary *, std::optional<ViewType>&, std::optional<WebExtensionTabIdentifier>&, std::optional<WebExtensionWindowIdentifier>&, NSString *sourceKey, NSString **outExceptionString);
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
