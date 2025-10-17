/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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
#if USE(APPLE_INTERNAL_SDK)

#import <BaseBoard/BSAction.h>
#import <BaseBoard/BSInvalidatable.h>

#else

@protocol BSInvalidatable <NSObject>
- (void)invalidate;
@end

@interface BSSettings : NSObject
- (id)objectForSetting:(NSUInteger)setting;
@end

@interface BSMutableSettings : BSSettings
- (void)setObject:(id)object forSetting:(NSUInteger)setting;
@end

@interface BSActionResponse : NSObject
+ (instancetype)response;
+ (instancetype)responseForError:(NSError *)error;
@property (nonatomic, retain, readonly) NSError *error;
@end

typedef void(^BSActionResponseHandler)(BSActionResponse *response);

@interface BSActionResponder : NSObject
+ (BSActionResponder *)responderWithHandler:(BSActionResponseHandler)handler;
@end

@interface BSAction : NSObject
- (instancetype)initWithInfo:(BSSettings *)info responder:(BSActionResponder *)responder;
- (BOOL)canSendResponse;
- (void)sendResponse:(BSActionResponse *)response;
@property (nonatomic, copy, readonly) BSSettings *info;
@end

#endif // USE(APPLE_INTERNAL_SDK)

typedef NS_ENUM(NSInteger, UIActionType) {
    UIActionTypeNotificationResponse = 26,
};

@class UNNotificationResponse;

@interface BSAction ()
@property (nonatomic, readonly) UIActionType UIActionType;
- (UNNotificationResponse *)response;
@end
