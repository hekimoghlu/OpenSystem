/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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

#include "JSWebExtensionAPIScripting.h"
#include "WebExtensionAPIObject.h"

OBJC_CLASS NSDictionary;
OBJC_CLASS NSObject;
OBJC_CLASS NSString;

namespace WebKit {

using FirstTimeRegistration = WebExtensionDynamicScripts::WebExtensionRegisteredScript::FirstTimeRegistration;

class WebExtension;

class WebExtensionAPIScripting : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIScripting, scripting, scripting);

public:
#if PLATFORM(COCOA)
    void executeScript(NSDictionary *, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void insertCSS(NSDictionary *, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void removeCSS(NSDictionary *, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void registerContentScripts(NSArray *, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getRegisteredContentScripts(NSDictionary *, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void updateContentScripts(NSArray *, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void unregisterContentScripts(NSDictionary *, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

private:
    friend class WebExtensionContext;

    static bool validateFilter(NSDictionary *filter, NSString **outExceptionString);

    static bool parseStyleLevel(NSDictionary *, NSString *key, std::optional<WebCore::UserStyleLevel>&, NSString **outExceptionString);
    static bool parseExecutionWorld(NSDictionary *, std::optional<WebExtensionContentWorldType>&, NSString **outExceptionString);
    static bool parseCSSInjectionOptions(NSDictionary *, WebExtensionScriptInjectionParameters&, NSString **outExceptionString);
    static bool parseTargetInjectionOptions(NSDictionary *, WebExtensionScriptInjectionParameters&, NSString **outExceptionString);
    static bool parseScriptInjectionOptions(NSDictionary *, WebExtensionScriptInjectionParameters&, NSString **outExceptionString);
    static bool parseRegisteredContentScripts(NSArray *, FirstTimeRegistration, Vector<WebExtensionRegisteredScriptParameters>&, NSString **outExceptionString);
#endif
};

NSArray *toWebAPI(const Vector<WebExtensionScriptInjectionResultParameters>&, bool returnExecutionResultOnly);
NSDictionary *toWebAPI(const WebExtensionRegisteredScriptParameters&);
NSString *toWebAPI(WebExtension::InjectionTime);

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
