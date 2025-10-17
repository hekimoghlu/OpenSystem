/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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

#include "JSWebExtensionAPIWindows.h"
#include "WebExtensionAPIObject.h"
#include "WebExtensionAPIWindowsEvent.h"
#include "WebExtensionWindow.h"
#include "WebExtensionWindowIdentifier.h"
#include "WebPageProxyIdentifier.h"

OBJC_CLASS NSDictionary;
OBJC_CLASS NSString;

namespace WebKit {

class WebPage;

class WebExtensionAPIWindows : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIWindows, windows, windows);

public:
#if PLATFORM(COCOA)
    using PopulateTabs = WebExtensionWindow::PopulateTabs;
    using WindowTypeFilter = WebExtensionWindow::TypeFilter;

    bool isPropertyAllowed(const ASCIILiteral& propertyName, WebPage*);

    void createWindow(NSDictionary *data, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void get(WebPageProxyIdentifier, double windowID, NSDictionary *info, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getCurrent(WebPageProxyIdentifier, NSDictionary *info, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getLastFocused(NSDictionary *info, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getAll(NSDictionary *info, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    void update(double windowID, NSDictionary *info, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void remove(double windowID, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    double windowIdentifierNone() const { return WebExtensionWindowConstants::None; }
    double windowIdentifierCurrent() const { return WebExtensionWindowConstants::Current; }

    WebExtensionAPIWindowsEvent& onCreated();
    WebExtensionAPIWindowsEvent& onRemoved();
    WebExtensionAPIWindowsEvent& onFocusChanged();

private:
    friend class WebExtensionAPITabs;
    friend class WebExtensionAPIWindowsEvent;

    static bool parsePopulateTabs(NSDictionary *, PopulateTabs&, NSString *sourceKey, NSString **outExceptionString);
    static bool parseWindowTypesFilter(NSDictionary *, OptionSet<WindowTypeFilter>&, NSString *sourceKey, NSString **outExceptionString);
    static bool parseWindowTypeFilter(NSString *, OptionSet<WindowTypeFilter>&, NSString *sourceKey, NSString **outExceptionString);
    static bool parseWindowGetOptions(NSDictionary *, PopulateTabs&, OptionSet<WindowTypeFilter>&, NSString *sourceKey, NSString **outExceptionString);
    bool parseWindowCreateOptions(NSDictionary *, WebExtensionWindowParameters&, NSString *sourceKey, NSString **outExceptionString);
    static bool parseWindowUpdateOptions(NSDictionary *, WebExtensionWindowParameters&, NSString *sourceKey, NSString **outExceptionString);

    RefPtr<WebExtensionAPIWindowsEvent> m_onCreated;
    RefPtr<WebExtensionAPIWindowsEvent> m_onRemoved;
    RefPtr<WebExtensionAPIWindowsEvent> m_onFocusChanged;
#endif
};

bool isValid(std::optional<WebExtensionWindowIdentifier>, NSString **outExceptionString);

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
