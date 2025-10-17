/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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

#import <ApplePushService/ApplePushService.h>

#else // if !USE(APPLE_INTERNAL_SDK)

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

WTF_EXTERN_C_BEGIN
extern NSString *const APSEnvironmentProduction;

typedef NS_ENUM(NSInteger, APSURLTokenErrorCode) {
    APSURLTokenErrorCodeInvalidArguments = 100,
    APSURLTokenErrorCodeXPCError = 101,
    APSURLTokenErrorCodeTopicSaltingFailed = 102,
    APSURLTokenErrorCodeTopicAlreadyInFilter = 103,
};
WTF_EXTERN_C_END

@class APSConnection;
@protocol APSConnectionDelegate;

@interface APSMessage : NSObject<NSCoding>
@property (nonatomic, retain) NSString *topic;
@property (nonatomic, retain) NSDictionary *userInfo;
@property (nonatomic, assign) NSUInteger identifier;
@end

@interface APSIncomingMessage : APSMessage
@end

#if HAVE(APPLE_PUSH_SERVICE_URL_TOKEN_SUPPORT)

@interface APSURLToken : NSObject<NSSecureCoding, NSCopying>
@property (nonatomic, strong) NSString *tokenURL;
@end

@interface APSURLTokenInfo : NSObject<NSSecureCoding, NSCopying>
- (instancetype)initWithTopic:(NSString *)topic vapidPublicKey:(nullable NSData *)vapidPublicKey;
@end

typedef void(^APSConnectionRequestURLTokenCompletion)(APSURLToken * __nullable token, NSError * __nullable error);
typedef void(^APSConnectionInvalidateURLTokenCompletion)(BOOL success, NSError * __nullable error);

#endif // HAVE(APPLE_PUSH_SERVICE_URL_TOKEN_SUPPORT)

@interface APSConnection : NSObject

- (instancetype)initWithEnvironmentName:(NSString *)environmentName namedDelegatePort:(nullable NSString *)namedDelegatePort queue:(dispatch_queue_t)queue;

#if HAVE(APPLE_PUSH_SERVICE_URL_TOKEN_SUPPORT)
- (void)requestURLTokenForInfo:(APSURLTokenInfo *)info completion:(APSConnectionRequestURLTokenCompletion)completion;
- (void)invalidateURLTokenForInfo:(APSURLTokenInfo *)info completion:(APSConnectionInvalidateURLTokenCompletion)completion;
#endif // HAVE(APPLE_PUSH_SERVICE_URL_TOKEN_SUPPORT)

@property (nonatomic, readwrite, assign, nullable) id<APSConnectionDelegate> delegate;

@property (nonatomic, readwrite, strong, setter=_setEnabledTopics:, nullable) NSArray<NSString *> *enabledTopics;
@property (nonatomic, readwrite, strong, setter=_setIgnoredTopics:, nullable) NSArray<NSString *> *ignoredTopics;
@property (nonatomic, readwrite, strong, setter=_setOpportunisticTopics:, nullable) NSArray<NSString *> *opportunisticTopics;
@property (nonatomic, readwrite, strong, setter=_setNonWakingTopics:, nullable) NSArray<NSString *> *nonWakingTopics;

- (void)setEnabledTopics:(NSArray<NSString *> *)enabledTopics ignoredTopics:(NSArray<NSString *> *)ignoredTopics opportunisticTopics:(NSArray<NSString *> *)opportunisticTopics nonWakingTopics:(NSArray<NSString *> *)nonWakingTopics;

@end

@protocol APSConnectionDelegate<NSObject>
- (void)connection:(APSConnection *)connection didReceivePublicToken:(NSData *)publicToken;
@optional
- (void)connection:(APSConnection *)connection didReceiveIncomingMessage:(APSIncomingMessage *)message;
@end

NS_ASSUME_NONNULL_END

#endif // !USE(APPLE_INTERNAL_SDK)
