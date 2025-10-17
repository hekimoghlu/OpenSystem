/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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
#if HAVE(APP_SSO)

#if USE(APPLE_INTERNAL_SDK)

#import <AppSSO/AppSSO.h>
#import <AppSSOCore/AppSSOCore.h>

#else

#if __has_include(<UIKit/UIKit.h>)
#import <UIKit/UIKit.h>
typedef UIViewController * SOAuthorizationViewController;
#elif __has_include(<AppKit/AppKit.h>)
typedef id SOAuthorizationViewController;
#endif

NS_ASSUME_NONNULL_BEGIN

#define kSOErrorAuthorizationPresentationFailed -7

extern NSErrorDomain const SOErrorDomain;

typedef NSString * SOAuthorizationOperation;

@class SOAuthorization;
@class SOAuthorizationParameters;

@protocol SOAuthorizationDelegate <NSObject>
@optional
- (void)authorizationDidNotHandle:(SOAuthorization *)authorization;
- (void)authorizationDidCancel:(SOAuthorization *)authorization;
- (void)authorizationDidComplete:(SOAuthorization *)authorization;
- (void)authorization:(SOAuthorization *)authorization didCompleteWithHTTPAuthorizationHeaders:(NSDictionary<NSString *, NSString *> *)httpAuthorizationHeaders;
- (void)authorization:(SOAuthorization *)authorization didCompleteWithHTTPResponse:(NSHTTPURLResponse *)httpResponse httpBody:(NSData *)httpBody;
- (void)authorization:(SOAuthorization *)authorization didCompleteWithError:(NSError *)error;
- (void)authorization:(SOAuthorization *)authorization presentViewController:(SOAuthorizationViewController)viewController withCompletion:(void (^)(BOOL success, NSError * _Nullable error))completion;
@end

typedef NSString * SOAuthorizationOption;
extern SOAuthorizationOption const SOAuthorizationOptionUserActionInitiated;
extern SOAuthorizationOption const SOAuthorizationOptionInitiatorOrigin;
extern SOAuthorizationOption const SOAuthorizationOptionInitiatingAction;

typedef NS_ENUM(NSInteger, SOAuthorizationInitiatingAction) {
    SOAuthorizationInitiatingActionRedirect,
    SOAuthorizationInitiatingActionPopUp,
    SOAuthorizationInitiatingActionKerberos,
    SOAuthorizationInitiatingActionSubframe,
};

@interface SOAuthorizationHintsCore : NSObject <NSSecureCoding>

- (instancetype)initWithLocalizedExtensionBundleDisplayName:(NSString *)localizedExtensionBundleDisplayName;

@property (nonatomic, readonly) NSString *localizedExtensionBundleDisplayName;

@end

@interface SOAuthorizationHints : NSObject

- (instancetype)initWithAuthorizationHintsCore:(SOAuthorizationHintsCore *)authorizationHintsCore;

@property (nonatomic, readonly) NSString *localizedExtensionBundleDisplayName;

@end

@interface SOAuthorization : NSObject
@property (weak) id<SOAuthorizationDelegate> delegate;
@property (retain, nullable) dispatch_queue_t delegateDispatchQueue;
@property (copy, nonatomic) NSDictionary *authorizationOptions;
@property (nonatomic) BOOL enableEmbeddedAuthorizationViewController;
- (void)getAuthorizationHintsWithURL:(NSURL *)url responseCode:(NSInteger)responseCode completion:(void (^)(SOAuthorizationHints * _Nullable authorizationHints, NSError * _Nullable error))completion;
+ (BOOL)canPerformAuthorizationWithURL:(NSURL *)url responseCode:(NSInteger)responseCode;
+ (BOOL)canPerformAuthorizationWithURL:(NSURL *)url responseCode:(NSInteger)responseCode useInternalExtensions:(BOOL)useInternalExtensions;
- (void)beginAuthorizationWithURL:(NSURL *)url httpHeaders:(NSDictionary <NSString *, NSString *> *)httpHeaders httpBody:(NSData *)httpBody;
- (void)beginAuthorizationWithOperation:(nullable SOAuthorizationOperation)operation url:(NSURL *)url httpHeaders:(NSDictionary <NSString *, NSString *> *)httpHeaders httpBody:(NSData *)httpBody;
- (void)beginAuthorizationWithParameters:(SOAuthorizationParameters *)parameters;
- (void)cancelAuthorization;
@end

NS_ASSUME_NONNULL_END

#endif // USE(APPLE_INTERNAL_SDK)

#endif // HAVE(APP_SSO)
