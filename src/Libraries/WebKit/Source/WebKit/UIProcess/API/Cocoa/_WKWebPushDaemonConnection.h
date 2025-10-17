/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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
#import <WebKit/WKFoundation.h>

typedef NS_ENUM(NSInteger, _WKWebPushPermissionState) {
    _WKWebPushPermissionStateDenied,
    _WKWebPushPermissionStateGranted,
    _WKWebPushPermissionStatePrompt,
};

@class WKSecurityOrigin;
@class _WKNotificationData;
@class _WKWebPushMessage;
@class _WKWebPushSubscriptionData;

NS_ASSUME_NONNULL_BEGIN

WK_EXTERN
@interface _WKWebPushDaemonConnectionConfiguration : NSObject

- (instancetype)init;
@property (nonatomic, copy) NSString *machServiceName;
@property (nonatomic, copy) NSString *partition;
@property (nonatomic, assign) audit_token_t hostApplicationAuditToken;
@property (nonatomic, copy) NSString *bundleIdentifierOverrideForTesting;
@end

WK_EXTERN
@interface _WKWebPushDaemonConnection : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithConfiguration:(_WKWebPushDaemonConnectionConfiguration *)configuration;

- (void)getPushPermissionStateForOrigin:(NSURL *)origin completionHandler:(void (^)(_WKWebPushPermissionState))completionHandler;
- (void)requestPushPermissionForOrigin:(NSURL *)origin completionHandler:(void (^)(BOOL))completionHandler;
- (void)setAppBadge:(NSUInteger *)badge origin:(NSURL *)origin;
- (void)subscribeToPushServiceForScope:(NSURL *)scopeURL applicationServerKey:(NSData *)key completionHandler:(void (^)(_WKWebPushSubscriptionData *, NSError *))completionHandler;
- (void)unsubscribeFromPushServiceForScope:(NSURL *)scopeURL completionHandler:(void (^)(BOOL unsubscribed, NSError *))completionHandler;
- (void)getSubscriptionForScope:(NSURL *)scopeURL completionHandler:(void (^)(_WKWebPushSubscriptionData *, NSError *))completionHandler;
- (void)getNextPendingPushMessage:(void (^)(_WKWebPushMessage *))completionHandler;
- (void)showNotification:(_WKNotificationData *)notificationData completionHandler:(void (^)())completionHandler;
- (void)getNotifications:(NSURL *)scopeURL tag:(NSString *)tag completionHandler:(void (^)(NSArray<_WKNotificationData *> *, NSError *))completionHandler;
- (void)cancelNotification:(NSURL *)securityOriginURL uuid:(NSUUID *)notificationIdentifier;

@end

NS_ASSUME_NONNULL_END
