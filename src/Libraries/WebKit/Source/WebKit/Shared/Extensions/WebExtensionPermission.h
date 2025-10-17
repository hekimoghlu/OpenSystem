/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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

#include <wtf/text/WTFString.h>

namespace WebKit {

/* Constants for specifying permission in a WebExtensionContext. */
class WebExtensionPermission {
public:
    static String activeTab();
    static String alarms();
    static String clipboardWrite();
    static String contextMenus();
    static String cookies();
    static String declarativeNetRequest();
    static String declarativeNetRequestFeedback();
    static String declarativeNetRequestWithHostAccess();
    static String menus();
    static String nativeMessaging();
    static String notifications();
    static String scripting();
#if ENABLE(WK_WEB_EXTENSION_SIDEBAR)
    static String sidePanel();
#endif // ENABLE(WK_WEB_EXTENSION_SIDEBAR)
    static String storage();
    static String tabs();
    static String unlimitedStorage();
    static String webNavigation();
    static String webRequest();
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
