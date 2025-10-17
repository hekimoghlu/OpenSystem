/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 7, 2022.
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
#import "UserNotificationsSPI.h"

#if HAVE(FULL_FEATURED_USER_NOTIFICATIONS)

@interface _WKMockUserNotificationCenter : NSObject
- (instancetype)initWithBundleIdentifier:(NSString *)bundleIdentifier;
- (void)addNotificationRequest:(UNNotificationRequest *)request withCompletionHandler:(void(^)(NSError *error))completionHandler;
- (void)getDeliveredNotificationsWithCompletionHandler:(void(^)(NSArray<UNNotification *> *notifications))completionHandler;
- (void)removePendingNotificationRequestsWithIdentifiers:(NSArray<NSString *> *) identifiers;
- (void)removeDeliveredNotificationsWithIdentifiers:(NSArray<NSString *> *) identifiers;
- (void)getNotificationSettingsWithCompletionHandler:(void(^)(UNNotificationSettings *settings))completionHandler;
- (void)requestAuthorizationWithOptions:(UNAuthorizationOptions)options completionHandler:(void (^)(BOOL granted, NSError *))completionHandler;
- (void)setNotificationCategories:(NSSet<UNNotificationCategory *> *) categories;
- (NSNumber *)getAppBadgeForTesting;
- (UNNotificationSettings *)notificationSettings;
@end

#endif
