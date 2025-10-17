/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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

NS_ASSUME_NONNULL_BEGIN

typedef NSUInteger UIBackgroundTaskIdentifier;

@class UNNotificationResponse;

WK_EXTERN NSString * const _WKWebPushActionTypePushEvent;
WK_EXTERN NSString * const _WKWebPushActionTypeNotificationClick;
WK_EXTERN NSString * const _WKWebPushActionTypeNotificationClose;

WK_EXTERN
@interface _WKWebPushAction : NSObject

- (instancetype)init NS_UNAVAILABLE;
+ (_WKWebPushAction *)webPushActionWithDictionary:(NSDictionary *)dictionary;
+ (_WKWebPushAction *)_webPushActionWithNotificationResponse:(UNNotificationResponse *)response;

@property (nonatomic, readonly) NSNumber *version;
@property (nonatomic, readonly) NSUUID *webClipIdentifier;
@property (nonatomic, readonly) NSString *type;

- (UIBackgroundTaskIdentifier)beginBackgroundTaskForHandling;

@end

NS_ASSUME_NONNULL_END
