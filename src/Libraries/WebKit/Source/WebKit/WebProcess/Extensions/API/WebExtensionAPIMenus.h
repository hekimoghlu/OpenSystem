/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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

#include "JSWebExtensionAPIMenus.h"
#include "WebExtensionAPIEvent.h"
#include "WebExtensionAPIObject.h"
#include "WebExtensionMenuItemParameters.h"

OBJC_CLASS NSDictionary;
OBJC_CLASS NSString;

namespace WebKit {

class WebExtensionAPIMenus : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIMenus, menus, menus);

public:
#if PLATFORM(COCOA)
    using ClickHandlerMap = HashMap<String, Ref<WebExtensionCallbackHandler>>;

    id createMenu(WebPage&, WebFrame&, NSDictionary *properties, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void update(WebPage&, WebFrame&, id identifier, NSDictionary *properties, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void remove(id identifier, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void removeAll(Ref<WebExtensionCallbackHandler>&&);

    WebExtensionAPIEvent& onClicked();

    double actionMenuTopLevelLimit() const { return webExtensionActionMenuItemTopLevelLimit; }

    const ClickHandlerMap& clickHandlers() const { return m_clickHandlerMap; }

private:
    enum class ForUpdate : bool { No, Yes };
    bool parseCreateAndUpdateProperties(ForUpdate, NSDictionary *, const URL& baseURL, std::optional<WebExtensionMenuItemParameters>&, RefPtr<WebExtensionCallbackHandler>&, NSString **outExceptionString);

    Markable<WebCore::FrameIdentifier> m_frameIdentifier;
    RefPtr<WebExtensionAPIEvent> m_onClicked;
    ClickHandlerMap m_clickHandlerMap;
#endif
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
