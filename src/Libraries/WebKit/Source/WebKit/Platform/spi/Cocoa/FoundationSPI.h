/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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

#if USE(APPLE_INTERNAL_SDK)

#import <Foundation/NSExtension.h>
#import <Foundation/NSLocale_Private.h>
#import <Foundation/NSPrivateDecls.h>

#if PLATFORM(IOS_FAMILY)
#import <Foundation/NSDistributedNotificationCenter.h>
#endif

#else

NS_ASSUME_NONNULL_BEGIN

@interface NSBundle ()
- (null_unspecified CFBundleRef)_cfBundle;
@end

#if PLATFORM(IOS_FAMILY)
@interface NSDistributedNotificationCenter : NSNotificationCenter
+ (NSDistributedNotificationCenter *)defaultCenter;
- (void)addObserver:(id)observer selector:(SEL)aSelector name:(nullable NSNotificationName)aName object:(nullable NSString *)anObject;
- (void)postNotificationName:(NSNotificationName)aName object:(nullable NSString *)anObject userInfo:(nullable NSDictionary *)aUserInfo;
@end
#endif

@interface NSExtension : NSObject
+ (NSExtension *)extensionWithIdentifier:(NSString *)bundleIdentifier error:(NSError **)error;
- (void)beginExtensionRequestWithInputItems:(NSArray *)inputItems completion:(void (^)(id <NSCopying>, NSError *))handler;
- (void)cancelExtensionRequestWithIdentifier:(id <NSCopying>)requestIdentifier;
@property (copy, NS_NONATOMIC_IOSONLY) void (^requestCompletionBlock)(id <NSCopying>, NSArray *);
@property (copy, NS_NONATOMIC_IOSONLY) void (^requestCancellationBlock)(id <NSCopying>, NSError *);
@property (copy, NS_NONATOMIC_IOSONLY) void (^requestInterruptionBlock)(id <NSCopying>);
@end

@interface NSLocale ()
+ (NSString *)_deviceLanguage;
@end

NS_ASSUME_NONNULL_END

#endif // USE(APPLE_INTERNAL_SDK)
