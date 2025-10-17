/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#import "WKWebExtensionPermissionPrivate.h"

WKWebExtensionPermission const WKWebExtensionPermissionActiveTab = @"activeTab";
WKWebExtensionPermission const WKWebExtensionPermissionAlarms = @"alarms";
WKWebExtensionPermission const WKWebExtensionPermissionClipboardWrite = @"clipboardWrite";
WKWebExtensionPermission const WKWebExtensionPermissionContextMenus = @"contextMenus";
WKWebExtensionPermission const WKWebExtensionPermissionCookies = @"cookies";
WKWebExtensionPermission const WKWebExtensionPermissionDeclarativeNetRequest = @"declarativeNetRequest";
WKWebExtensionPermission const WKWebExtensionPermissionDeclarativeNetRequestFeedback = @"declarativeNetRequestFeedback";
WKWebExtensionPermission const WKWebExtensionPermissionDeclarativeNetRequestWithHostAccess = @"declarativeNetRequestWithHostAccess";
WKWebExtensionPermission const WKWebExtensionPermissionMenus = @"menus";
WKWebExtensionPermission const WKWebExtensionPermissionNativeMessaging = @"nativeMessaging";
WKWebExtensionPermission const WKWebExtensionPermissionNotifications = @"notifications";
WKWebExtensionPermission const WKWebExtensionPermissionScripting = @"scripting";
WKWebExtensionPermission const WKWebExtensionPermissionSidePanel = @"sidePanel";
WKWebExtensionPermission const WKWebExtensionPermissionStorage = @"storage";
WKWebExtensionPermission const WKWebExtensionPermissionTabs = @"tabs";
WKWebExtensionPermission const WKWebExtensionPermissionUnlimitedStorage = @"unlimitedStorage";
WKWebExtensionPermission const WKWebExtensionPermissionWebNavigation = @"webNavigation";
WKWebExtensionPermission const WKWebExtensionPermissionWebRequest = @"webRequest";
