/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#include "config.h"
#include "WebExtensionPermission.h"

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

String WebExtensionPermission::activeTab()
{
    return "activeTab"_s;
}

String WebExtensionPermission::alarms()
{
    return "alarms"_s;
}

String WebExtensionPermission::clipboardWrite()
{
    return "clipboardWrite"_s;
}

String WebExtensionPermission::contextMenus()
{
    return "contextMenus"_s;
}

String WebExtensionPermission::cookies()
{
    return "cookies"_s;
}

String WebExtensionPermission::declarativeNetRequest()
{
    return "declarativeNetRequest"_s;
}

String WebExtensionPermission::declarativeNetRequestFeedback()
{
    return "declarativeNetRequestFeedback"_s;
}

String WebExtensionPermission::declarativeNetRequestWithHostAccess()
{
    return "declarativeNetRequestWithHostAccess"_s;
}

String WebExtensionPermission::menus()
{
    return "menus"_s;
}

String WebExtensionPermission::nativeMessaging()
{
    return "nativeMessaging"_s;
}

String WebExtensionPermission::notifications()
{
    return "notifications"_s;
}

String WebExtensionPermission::scripting()
{
    return "scripting"_s;
}

#if ENABLE(WK_WEB_EXTENSION_SIDEBAR)
String WebExtensionPermission::sidePanel()
{
    return "sidePanel"_s;
}
#endif // ENABLE(WK_WEB_EXTENSION_SIDEBAR)

String WebExtensionPermission::storage()
{
    return "storage"_s;
}

String WebExtensionPermission::tabs()
{
    return "tabs"_s;
}

String WebExtensionPermission::unlimitedStorage()
{
    return "unlimitedStorage"_s;
}

String WebExtensionPermission::webNavigation()
{
    return "webNavigation"_s;
}

String WebExtensionPermission::webRequest()
{
    return "webRequest"_s;
}

} // namespace WebKit

#endif
