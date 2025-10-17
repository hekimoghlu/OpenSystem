/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
#pragma once

#if ENABLE(APPLE_PAY_AMS_UI)

#if USE(APPLE_INTERNAL_SDK)

#import <AppleMediaServices/AMSBag.h>
#import <AppleMediaServices/AMSBagConsumer.h>
#import <AppleMediaServices/AMSBagProtocol.h>
#import <AppleMediaServices/AMSEngagementRequest.h>
#import <AppleMediaServices/AMSEngagementResult.h>
#import <AppleMediaServices/AMSPromise.h>
#import <AppleMediaServices/AMSTask.h>

#else // if !USE(APPLE_INTERNAL_SDK)

NS_ASSUME_NONNULL_BEGIN

@interface AMSPromise<ResultType> : NSObject
- (void)addFinishBlock:(void(^)(ResultType _Nullable result, NSError * _Nullable error))finishBlock;
@end

@protocol AMSBagProtocol <NSObject>
@end

@interface AMSBag : NSObject <AMSBagProtocol>
@end

@protocol AMSBagConsumer <NSObject>
@optional
+ (AMSBag *)createBagForSubProfile;
@end

@interface AMSEngagementRequest : NSObject <NSSecureCoding>
@end

@interface AMSEngagementResult : NSObject <NSSecureCoding>
@end

@interface AMSTask : NSObject
- (BOOL)cancel;
@end

NS_ASSUME_NONNULL_END

#endif // !USE(APPLE_INTERNAL_SDK)

NS_ASSUME_NONNULL_BEGIN

@interface AMSEngagementRequest (Staging_84159382)
- (instancetype)initWithRequestDictionary:(NSDictionary *)dictionary;
@property (NS_NONATOMIC_IOSONLY, strong, nullable) NSURL *originatingURL;
@end

NS_ASSUME_NONNULL_END

#endif // ENABLE(APPLE_PAY_AMS_UI)
