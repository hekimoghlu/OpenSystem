/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionAPINamespace.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "CocoaHelpers.h"
#import "WKWebExtensionPermission.h"
#import "WebExtensionControllerProxy.h"

#if ENABLE(INSPECTOR_EXTENSIONS) || ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)
#import "WebPage.h"
#endif

namespace WebKit {

bool WebExtensionAPINamespace::isPropertyAllowed(const ASCIILiteral& name, WebPage* page)
{
    if (UNLIKELY(extensionContext().isUnsupportedAPI(propertyPath(), name)))
        return false;

    if (name == "action"_s)
        return extensionContext().supportsManifestVersion(3) && objectForKey<NSDictionary>(extensionContext().manifest(), @"action", false);

    if (name == "commands"_s)
        return objectForKey<NSDictionary>(extensionContext().manifest(), @"commands", false);

    if (name == "declarativeNetRequest"_s)
        return extensionContext().hasPermission(name) || extensionContext().hasPermission("declarativeNetRequestWithHostAccess"_s);

    if (name == "browserAction"_s)
        return !extensionContext().supportsManifestVersion(3) && objectForKey<NSDictionary>(extensionContext().manifest(), @"browser_action", false);

#if ENABLE(INSPECTOR_EXTENSIONS)
    if (name == "devtools"_s)
        return objectForKey<NSString>(extensionContext().manifest(), @"devtools_page") && page && (page->isInspectorPage() || extensionContext().isInspectorBackgroundPage(*page));
#else
    if (name == "devtools"_s)
        return false;
#endif

    if (name == "notifications"_s) {
        // FIXME: <rdar://problem/57202210> Add support for browser.notifications.
        // Notifications are currently only available in test mode as an empty stub.
        if (!extensionContext().inTestingMode())
            return false;
        goto finish;
    }

    if (name == "pageAction"_s)
        return !extensionContext().supportsManifestVersion(3) && objectForKey<NSDictionary>(extensionContext().manifest(), @"page_action", false);

#if ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)
    // If the extension requests both sidePanel and sidebarAction, we will give them sidebarAction --
    // we check in sidePanel that there is no sidebar_action key, but we do not check in sidebarAction
    // that there is no sidePanel permission
    if (name == "sidePanel"_s)
        return page->corePage()->settings().webExtensionSidebarEnabled() && extensionContext().hasPermission("sidePanel"_s) && !objectForKey<NSDictionary>(extensionContext().manifest(), @"sidebar_action", true);
    if (name == "sidebarAction"_s)
        return page->corePage()->settings().webExtensionSidebarEnabled() && objectForKey<NSDictionary>(extensionContext().manifest(), @"sidebar_action", true);
#endif // ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)

    if (name == "storage"_s)
        return extensionContext().hasPermission(name) || extensionContext().hasPermission("unlimitedStorage"_s);

    if (name == "test"_s)
        return extensionContext().inTestingMode();

finish:
    // The rest of the property names marked dynamic in WebExtensionAPINamespace.idl match permission names.
    // Check for the permission to determine if the property is allowed to be accessed.
    return extensionContext().hasPermission(name);
}

WebExtensionAPIAction& WebExtensionAPINamespace::action()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/action

    if (!m_action)
        m_action = WebExtensionAPIAction::create(*this);

    return *m_action;
}

WebExtensionAPIAlarms& WebExtensionAPINamespace::alarms()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/alarms

    if (!m_alarms)
        m_alarms = WebExtensionAPIAlarms::create(*this);

    return *m_alarms;
}

WebExtensionAPICommands& WebExtensionAPINamespace::commands()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/commands

    if (!m_commands)
        m_commands = WebExtensionAPICommands::create(*this);

    return *m_commands;
}

WebExtensionAPICookies& WebExtensionAPINamespace::cookies()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/cookies

    if (!m_cookies)
        m_cookies = WebExtensionAPICookies::create(*this);

    return *m_cookies;
}

WebExtensionAPIDeclarativeNetRequest& WebExtensionAPINamespace::declarativeNetRequest()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/declarativeNetRequest

    if (!m_declarativeNetRequest)
        m_declarativeNetRequest = WebExtensionAPIDeclarativeNetRequest::create(*this);

    return *m_declarativeNetRequest;
}

#if ENABLE(INSPECTOR_EXTENSIONS)
WebExtensionAPIDevTools& WebExtensionAPINamespace::devtools()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools

    if (!m_devtools)
        m_devtools = WebExtensionAPIDevTools::create(*this);

    return *m_devtools;
}
#endif // ENABLE(INSPECTOR_EXTENSIONS)

WebExtensionAPIExtension& WebExtensionAPINamespace::extension()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/extension

    if (!m_extension)
        m_extension = WebExtensionAPIExtension::create(*this);

    return *m_extension;
}

WebExtensionAPILocalization& WebExtensionAPINamespace::i18n()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/i18n

    if (!m_i18n)
        m_i18n = WebExtensionAPILocalization::create(*this);

    return *m_i18n;
}

WebExtensionAPIMenus& WebExtensionAPINamespace::menus()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/menus

    if (!m_menus)
        m_menus = WebExtensionAPIMenus::create(*this);

    return *m_menus;
}

WebExtensionAPINotifications& WebExtensionAPINamespace::notifications()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/notifications

    if (!m_notifications)
        m_notifications = WebExtensionAPINotifications::create(*this);

    return *m_notifications;
}

WebExtensionAPIPermissions& WebExtensionAPINamespace::permissions()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/permissions

    if (!m_permissions)
        m_permissions = WebExtensionAPIPermissions::create(*this);

    return *m_permissions;
}

WebExtensionAPIRuntime& WebExtensionAPINamespace::runtime() const
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/runtime

    if (!m_runtime) {
        m_runtime = WebExtensionAPIRuntime::create(contentWorldType(), extensionContext());
        m_runtime->setPropertyPath("runtime"_s, this);
    }

    return *m_runtime;
}

WebExtensionAPIScripting& WebExtensionAPINamespace::scripting()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/scripting

    if (!m_scripting)
        m_scripting = WebExtensionAPIScripting::create(*this);

    return *m_scripting;
}

#if ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)
WebExtensionAPISidePanel& WebExtensionAPINamespace::sidePanel()
{
    // Documentation: https://developer.chrome.com/docs/extensions/reference/api/sidePanel

    if (!m_sidePanel)
        m_sidePanel = WebExtensionAPISidePanel::create(*this);

    return *m_sidePanel;
}

WebExtensionAPISidebarAction& WebExtensionAPINamespace::sidebarAction()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/sidebarAction

    if (!m_sidebarAction)
        m_sidebarAction = WebExtensionAPISidebarAction::create(*this);

    return *m_sidebarAction;
}
#endif

WebExtensionAPIStorage& WebExtensionAPINamespace::storage()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/storage

    if (!m_storage)
        m_storage = WebExtensionAPIStorage::create(*this);

    return *m_storage;
}

WebExtensionAPITabs& WebExtensionAPINamespace::tabs()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/tabs

    if (!m_tabs)
        m_tabs = WebExtensionAPITabs::create(*this);

    return *m_tabs;
}

WebExtensionAPITest& WebExtensionAPINamespace::test()
{
    // Documentation: None (Testing Only)

    if (!m_test)
        m_test = WebExtensionAPITest::create(*this);

    return *m_test;
}

WebExtensionAPIWindows& WebExtensionAPINamespace::windows()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/windows

    if (!m_windows)
        m_windows = WebExtensionAPIWindows::create(*this);

    return *m_windows;
}

WebExtensionAPIWebNavigation& WebExtensionAPINamespace::webNavigation()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/webNavigation

    if (!m_webNavigation)
        m_webNavigation = WebExtensionAPIWebNavigation::create(*this);

    return *m_webNavigation;
}

WebExtensionAPIWebRequest& WebExtensionAPINamespace::webRequest()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/webRequest

    if (!m_webRequest)
        m_webRequest = WebExtensionAPIWebRequest::create(*this);

    return *m_webRequest;
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
