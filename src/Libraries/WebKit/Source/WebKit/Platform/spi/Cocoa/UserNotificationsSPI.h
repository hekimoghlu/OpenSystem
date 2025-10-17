/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#import <UserNotifications/UserNotifications.h>

#if USE(APPLE_INTERNAL_SDK)

#import <UserNotifications/UNNotificationContent_Private.h>
#import <UserNotifications/UNNotificationSettings_Private.h>
#import <UserNotifications/UNUserNotificationCenter_Private.h>

#if HAVE(FULL_FEATURED_USER_NOTIFICATIONS)
#import <UserNotifications/UNNotificationIcon.h>
#import <UserNotifications/UNNotificationIcon_Private.h>
#import <UserNotifications/UNNotification_Private.h>
#endif

#else // USE(APPLE_INTERNAL_SDK)

#if HAVE(FULL_FEATURED_USER_NOTIFICATIONS)
@interface UNNotification ()
+ (instancetype)notificationWithRequest:(UNNotificationRequest *)request date:(NSDate *)date;

@property (readonly) NSString *sourceIdentifier;

@end

@interface UNNotificationIcon : NSObject <NSCopying, NSSecureCoding>
+ (instancetype)iconForApplicationIdentifier:(NSString *)applicationIdentifier;
@end
#endif

@interface UNMutableNotificationContent ()
@property (NS_NONATOMIC_IOSONLY, copy) NSString *defaultActionBundleIdentifier;
#if HAVE(FULL_FEATURED_USER_NOTIFICATIONS)
@property (NS_NONATOMIC_IOSONLY, copy) UNNotificationIcon *icon;
#endif
@end

@interface UNMutableNotificationSettings : UNNotificationSettings
+ (instancetype)emptySettings;
@property (NS_NONATOMIC_IOSONLY, readwrite) UNAuthorizationStatus authorizationStatus;
@end

@interface UNUserNotificationCenter ()
- (instancetype)initWithBundleIdentifier:(NSString *)bundleIdentifier;
- (UNNotificationSettings *)notificationSettings;
@end

#endif // USE(APPLE_INTERNAL_SDK)

