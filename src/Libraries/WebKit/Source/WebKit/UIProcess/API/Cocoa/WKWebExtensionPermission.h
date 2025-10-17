/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#import <Foundation/Foundation.h>
#import <WebKit/WKFoundation.h>

/*! @abstract Constants for specifying permission in a ``WKWebExtensionContext``. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
typedef NSString * WKWebExtensionPermission NS_TYPED_EXTENSIBLE_ENUM NS_SWIFT_NAME(WKWebExtension.Permission);

/*! @abstract The `activeTab` permission requests that when the user interacts with the extension, the extension is granted extra permissions for the active tab only. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionActiveTab NS_SWIFT_NONISOLATED;

/*! @abstract The `alarms` permission requests access to the `browser.alarms` APIs. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionAlarms NS_SWIFT_NONISOLATED;

/*! @abstract The `clipboardWrite` permission requests access to write to the clipboard. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionClipboardWrite NS_SWIFT_NONISOLATED;

/*! @abstract The `contextMenus` permission requests access to the `browser.contextMenus` APIs. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionContextMenus NS_SWIFT_NONISOLATED;

/*! @abstract The `cookies` permission requests access to the `browser.cookies` APIs. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionCookies NS_SWIFT_NONISOLATED;

/*! @abstract The `declarativeNetRequest` permission requests access to the `browser.declarativeNetRequest` APIs. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionDeclarativeNetRequest NS_SWIFT_NONISOLATED;

/*! @abstract The `declarativeNetRequestFeedback` permission requests access to the `browser.declarativeNetRequest` APIs with extra information on matched rules. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionDeclarativeNetRequestFeedback NS_SWIFT_NONISOLATED;

/*! @abstract The `declarativeNetRequestWithHostAccess` permission requests access to the `browser.declarativeNetRequest` APIs with the ability to modify or redirect requests. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionDeclarativeNetRequestWithHostAccess NS_SWIFT_NONISOLATED;

/*! @abstract The `menus` permission requests access to the `browser.menus` APIs. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionMenus NS_SWIFT_NONISOLATED;

/*! @abstract The `nativeMessaging` permission requests access to send messages to the App Extension bundle. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionNativeMessaging NS_SWIFT_NONISOLATED;

/*! @abstract The `scripting` permission requests access to the `browser.scripting` APIs. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionScripting NS_SWIFT_NONISOLATED;

/*! @abstract The `storage` permission requests access to the `browser.storage` APIs. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionStorage NS_SWIFT_NONISOLATED;

/*! @abstract The `tabs` permission requests access extra information on the `browser.tabs` APIs. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionTabs NS_SWIFT_NONISOLATED;

/*! @abstract The `unlimitedStorage` permission requests access to an unlimited quota on the `browser.storage.local` APIs. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionUnlimitedStorage NS_SWIFT_NONISOLATED;

/*! @abstract The `webNavigation` permission requests access to the `browser.webNavigation` APIs. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionWebNavigation NS_SWIFT_NONISOLATED;

/*! @abstract The `webRequest` permission requests access to the `browser.webRequest` APIs. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionPermission const WKWebExtensionPermissionWebRequest NS_SWIFT_NONISOLATED;
