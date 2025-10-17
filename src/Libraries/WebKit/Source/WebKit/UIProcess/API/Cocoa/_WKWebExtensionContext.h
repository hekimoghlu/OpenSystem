/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
#import <WebKit/WKWebExtensionContextPrivate.h>

#define HAVE_WK_WEB_EXTENSION_CONTEXT_ARRAY_BASED_DID_SELECT_TABS 1

WK_EXTERN
@interface _WKWebExtensionContext : WKWebExtensionContext
@end

#define _WKWebExtensionContext WKWebExtensionContext

#define _WKWebExtensionContextPermissionStatus WKWebExtensionContextPermissionStatus
#define _WKWebExtensionContextPermissionStatusDeniedExplicitly WKWebExtensionContextPermissionStatusDeniedExplicitly
#define _WKWebExtensionContextPermissionStatusDeniedImplicitly WKWebExtensionContextPermissionStatusDeniedImplicitly
#define _WKWebExtensionContextPermissionStatusRequestedImplicitly WKWebExtensionContextPermissionStatusRequestedImplicitly
#define _WKWebExtensionContextPermissionStatusUnknown WKWebExtensionContextPermissionStatusUnknown
#define _WKWebExtensionContextPermissionStatusRequestedExplicitly WKWebExtensionContextPermissionStatusRequestedExplicitly
#define _WKWebExtensionContextPermissionStatusGrantedImplicitly WKWebExtensionContextPermissionStatusGrantedImplicitly
#define _WKWebExtensionContextPermissionStatusGrantedExplicitly WKWebExtensionContextPermissionStatusGrantedExplicitly

#define _WKWebExtensionPermission WKWebExtensionPermission

WK_EXTERN NSErrorDomain const _WKWebExtensionContextErrorDomain;

WK_EXTERN NSNotificationName const _WKWebExtensionContextPermissionsWereGrantedNotification;
WK_EXTERN NSNotificationName const _WKWebExtensionContextPermissionsWereDeniedNotification;
WK_EXTERN NSNotificationName const _WKWebExtensionContextGrantedPermissionsWereRemovedNotification;
WK_EXTERN NSNotificationName const _WKWebExtensionContextDeniedPermissionsWereRemovedNotification;

WK_EXTERN NSNotificationName const _WKWebExtensionContextPermissionMatchPatternsWereGrantedNotification;
WK_EXTERN NSNotificationName const _WKWebExtensionContextPermissionMatchPatternsWereDeniedNotification;
WK_EXTERN NSNotificationName const _WKWebExtensionContextGrantedPermissionMatchPatternsWereRemovedNotification;
WK_EXTERN NSNotificationName const _WKWebExtensionContextDeniedPermissionMatchPatternsWereRemovedNotification;

WK_EXTERN NSString * const _WKWebExtensionContextNotificationUserInfoKeyPermissions;
WK_EXTERN NSString * const _WKWebExtensionContextNotificationUserInfoKeyMatchPatterns;
