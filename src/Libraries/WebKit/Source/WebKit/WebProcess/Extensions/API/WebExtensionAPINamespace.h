/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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

#include "JSWebExtensionAPINamespace.h"
#include "WebExtensionAPIAction.h"
#include "WebExtensionAPIAlarms.h"
#include "WebExtensionAPICommands.h"
#include "WebExtensionAPICookies.h"
#include "WebExtensionAPIDeclarativeNetRequest.h"
#include "WebExtensionAPIDevTools.h"
#include "WebExtensionAPIExtension.h"
#include "WebExtensionAPILocalization.h"
#include "WebExtensionAPIMenus.h"
#include "WebExtensionAPINotifications.h"
#include "WebExtensionAPIObject.h"
#include "WebExtensionAPIPermissions.h"
#include "WebExtensionAPIRuntime.h"
#include "WebExtensionAPIScripting.h"
#include "WebExtensionAPISidePanel.h"
#include "WebExtensionAPISidebarAction.h"
#include "WebExtensionAPIStorage.h"
#include "WebExtensionAPITabs.h"
#include "WebExtensionAPITest.h"
#include "WebExtensionAPIWebNavigation.h"
#include "WebExtensionAPIWebRequest.h"
#include "WebExtensionAPIWindows.h"

namespace WebKit {

class WebExtensionAPIExtension;
class WebExtensionAPIRuntime;

class WebExtensionAPINamespace : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPINamespace, namespace, browser);

public:
#if PLATFORM(COCOA)
    bool isPropertyAllowed(const ASCIILiteral& propertyName, WebPage*);

    WebExtensionAPIAction& action();
    WebExtensionAPIAlarms& alarms();
    WebExtensionAPIAction& browserAction() { return action(); }
    WebExtensionAPICommands& commands();
    WebExtensionAPICookies& cookies();
    WebExtensionAPIMenus& contextMenus() { return menus(); }
    WebExtensionAPIDeclarativeNetRequest& declarativeNetRequest();
#if ENABLE(INSPECTOR_EXTENSIONS)
    WebExtensionAPIDevTools& devtools();
#endif
    WebExtensionAPIExtension& extension();
    WebExtensionAPILocalization& i18n();
    WebExtensionAPIMenus& menus();
    WebExtensionAPINotifications& notifications();
    WebExtensionAPIAction& pageAction() { return action(); }
    WebExtensionAPIPermissions& permissions();
    WebExtensionAPIRuntime& runtime() const final;
    WebExtensionAPIScripting& scripting();
#if ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)
    WebExtensionAPISidePanel& sidePanel();
    WebExtensionAPISidebarAction& sidebarAction();
#endif
    WebExtensionAPIStorage& storage();
    WebExtensionAPITabs& tabs();
    WebExtensionAPITest& test();
    WebExtensionAPIWindows& windows();
    WebExtensionAPIWebNavigation& webNavigation();
    WebExtensionAPIWebRequest& webRequest();
#endif

private:
    RefPtr<WebExtensionAPIAction> m_action;
    RefPtr<WebExtensionAPIAlarms> m_alarms;
    RefPtr<WebExtensionAPICommands> m_commands;
    RefPtr<WebExtensionAPICookies> m_cookies;
    RefPtr<WebExtensionAPIDeclarativeNetRequest> m_declarativeNetRequest;
#if ENABLE(INSPECTOR_EXTENSIONS)
    RefPtr<WebExtensionAPIDevTools> m_devtools;
#endif
    RefPtr<WebExtensionAPIExtension> m_extension;
    RefPtr<WebExtensionAPILocalization> m_i18n;
    RefPtr<WebExtensionAPIMenus> m_menus;
    RefPtr<WebExtensionAPINotifications> m_notifications;
    RefPtr<WebExtensionAPIPermissions> m_permissions;
    mutable RefPtr<WebExtensionAPIRuntime> m_runtime;
    RefPtr<WebExtensionAPIScripting> m_scripting;
#if ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)
    RefPtr<WebExtensionAPISidePanel> m_sidePanel;
    RefPtr<WebExtensionAPISidebarAction> m_sidebarAction;
#endif
    RefPtr<WebExtensionAPIStorage> m_storage;
    RefPtr<WebExtensionAPITabs> m_tabs;
    RefPtr<WebExtensionAPITest> m_test;
    RefPtr<WebExtensionAPIWindows> m_windows;
    RefPtr<WebExtensionAPIWebNavigation> m_webNavigation;
    RefPtr<WebExtensionAPIWebRequest> m_webRequest;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
