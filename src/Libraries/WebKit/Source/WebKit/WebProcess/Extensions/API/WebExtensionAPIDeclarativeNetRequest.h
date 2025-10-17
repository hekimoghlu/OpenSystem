/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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

#include "JSWebExtensionAPIDeclarativeNetRequest.h"
#include "WebExtensionAPIObject.h"
#include "WebExtensionConstants.h"

namespace WebKit {

class WebPage;

class WebExtensionAPIDeclarativeNetRequest : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIDeclarativeNetRequest, declarativeNetRequest, declarativeNetRequest);

public:
#if PLATFORM(COCOA)
    void updateEnabledRulesets(NSDictionary *options, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getEnabledRulesets(Ref<WebExtensionCallbackHandler>&&);

    void updateDynamicRules(NSDictionary *options, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getDynamicRules(NSDictionary *filter, Ref<WebExtensionCallbackHandler>&&);

    void updateSessionRules(NSDictionary *options, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getSessionRules(NSDictionary *filter, Ref<WebExtensionCallbackHandler>&&);

    void getMatchedRules(NSDictionary *filter, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void isRegexSupported(NSDictionary *options, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void setExtensionActionOptions(NSDictionary *options, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    double maxNumberOfStaticRulesets() const { return webExtensionDeclarativeNetRequestMaximumNumberOfStaticRulesets; }
    double maxNumberOfEnabledRulesets() const { return webExtensionDeclarativeNetRequestMaximumNumberOfEnabledRulesets; }
    double maxNumberOfDynamicAndSessionRules() const { return webExtensionDeclarativeNetRequestMaximumNumberOfDynamicAndSessionRules; }
#endif
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
