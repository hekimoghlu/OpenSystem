/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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
#import "config.h"
#import "_WKWebExtension.h"

#import "_WKWebExtensionAction.h"
#import "_WKWebExtensionCommand.h"
#import "_WKWebExtensionContext.h"
#import "_WKWebExtensionController.h"
#import "_WKWebExtensionControllerConfiguration.h"
#import "_WKWebExtensionDataRecord.h"
#import "_WKWebExtensionMatchPattern.h"
#import "_WKWebExtensionMessagePort.h"
#import "_WKWebExtensionTab.h"
#import "_WKWebExtensionTabCreationOptions.h"
#import "_WKWebExtensionWindow.h"
#import "_WKWebExtensionWindowCreationOptions.h"

#undef _WKWebExtensionTab
#undef _WKWebExtensionWindow
@interface _WKWebExtensionStaging : NSObject <_WKWebExtensionTab, _WKWebExtensionWindow>
@end

@implementation _WKWebExtensionStaging
@end

NSErrorDomain const _WKWebExtensionErrorDomain = @"WKWebExtensionErrorDomain";

NSNotificationName const _WKWebExtensionErrorsWereUpdatedNotification = @"WKWebExtensionContextErrorsDidUpdate";

#undef _WKWebExtension
@implementation _WKWebExtension
@end

NSNotificationName const _WKWebExtensionActionPropertiesDidChangeNotification = @"WKWebExtensionActionPropertiesDidChange";

#undef _WKWebExtensionAction
@implementation _WKWebExtensionAction
@end

#undef _WKWebExtensionCommand
@implementation _WKWebExtensionCommand
@end

NSErrorDomain const _WKWebExtensionContextErrorDomain = @"WKWebExtensionContextErrorDomain";

NSNotificationName const _WKWebExtensionContextPermissionsWereGrantedNotification = @"WKWebExtensionContextPermissionsWereGranted";
NSNotificationName const _WKWebExtensionContextPermissionsWereDeniedNotification = @"WKWebExtensionContextPermissionsWereDenied";
NSNotificationName const _WKWebExtensionContextGrantedPermissionsWereRemovedNotification = @"WKWebExtensionContextGrantedPermissionsWereRemoved";
NSNotificationName const _WKWebExtensionContextDeniedPermissionsWereRemovedNotification = @"WKWebExtensionContextDeniedPermissionsWereRemoved";

NSNotificationName const _WKWebExtensionContextPermissionMatchPatternsWereGrantedNotification = @"WKWebExtensionContextPermissionMatchPatternsWereGranted";
NSNotificationName const _WKWebExtensionContextPermissionMatchPatternsWereDeniedNotification = @"WKWebExtensionContextPermissionMatchPatternsWereDenied";
NSNotificationName const _WKWebExtensionContextGrantedPermissionMatchPatternsWereRemovedNotification = @"WKWebExtensionContextGrantedPermissionMatchPatternsWereRemoved";
NSNotificationName const _WKWebExtensionContextDeniedPermissionMatchPatternsWereRemovedNotification = @"WKWebExtensionContextDeniedPermissionMatchPatternsWereRemoved";

NSString * const _WKWebExtensionContextNotificationUserInfoKeyPermissions = @"permissions";
NSString * const _WKWebExtensionContextNotificationUserInfoKeyMatchPatterns = @"matchPatterns";

#undef _WKWebExtensionContext
@implementation _WKWebExtensionContext
@end

#undef _WKWebExtensionController
@implementation _WKWebExtensionController
@end

#undef _WKWebExtensionControllerConfiguration
@implementation _WKWebExtensionControllerConfiguration
@end

NSErrorDomain const _WKWebExtensionDataRecordErrorDomain = @"WKWebExtensionDataRecordErrorDomain";

#undef _WKWebExtensionDataRecord
@implementation _WKWebExtensionDataRecord
@end

NSErrorDomain const _WKWebExtensionMatchPatternErrorDomain = @"WKWebExtensionMatchPatternErrorDomain";

#undef _WKWebExtensionMatchPattern
@implementation _WKWebExtensionMatchPattern
@end

NSErrorDomain const _WKWebExtensionMessagePortErrorDomain = @"WKWebExtensionMessagePortErrorDomain";

#undef _WKWebExtensionMessagePort
@implementation _WKWebExtensionMessagePort
@end

#undef _WKWebExtensionTabCreationOptions
@implementation _WKWebExtensionTabCreationOptions
@end

#undef _WKWebExtensionWindowCreationOptions
@implementation _WKWebExtensionWindowCreationOptions
@end

