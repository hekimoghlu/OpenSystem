/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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

#include "JSWebExtensionAPICookies.h"
#include "WebExtensionAPIEvent.h"
#include "WebExtensionAPIObject.h"

OBJC_CLASS NSDictionary;
OBJC_CLASS NSString;

namespace WebKit {

class WebExtensionAPICookies : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPICookies, cookies, cookies);

public:
#if PLATFORM(COCOA)
    void get(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getAll(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void set(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void remove(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void getAllCookieStores(Ref<WebExtensionCallbackHandler>&&);

    WebExtensionAPIEvent& onChanged();

private:
    using ParsedDetails = std::tuple<std::optional<PAL::SessionID>, String, URL>;
    static std::optional<ParsedDetails> parseCookieDetails(NSDictionary *details, NSArray *requiredKeys, NSString **outExceptionString);

    RefPtr<WebExtensionAPIEvent> m_onChanged;
#endif
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
