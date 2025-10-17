/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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

#import <SpringBoardServices/SBSStatusBarStyleOverridesAssertion.h>
#import <SpringBoardServices/SBSStatusBarStyleOverridesCoordinator.h>

#else

#import <Foundation/Foundation.h>

typedef enum _UIStatusBarStyleOverrides : uint32_t {
    UIStatusBarStyleOverrideWebRTCAudioCapture  = 1 << 24,
    UIStatusBarStyleOverrideWebRTCCapture       = 1 << 25
} UIStatusBarStyleOverrides;

typedef void (^SBSStatusBarStyleOverridesAssertionAcquisitionHandler)(BOOL acquired);

@interface SBSStatusBarStyleOverridesAssertion : NSObject
@property (nonatomic, readonly) UIStatusBarStyleOverrides statusBarStyleOverrides;
@property (nonatomic, readwrite, copy) NSString *statusString;
+ (instancetype)assertionWithStatusBarStyleOverrides:(UIStatusBarStyleOverrides)overrides forPID:(pid_t)pid exclusive:(BOOL)exclusive showsWhenForeground:(BOOL)showsWhenForeground;
- (instancetype)initWithStatusBarStyleOverrides:(UIStatusBarStyleOverrides)overrides forPID:(pid_t)pid exclusive:(BOOL)exclusive showsWhenForeground:(BOOL)showsWhenForeground;
- (void)acquireWithHandler:(SBSStatusBarStyleOverridesAssertionAcquisitionHandler)handler invalidationHandler:(void (^)(void))invalidationHandler;
- (void)invalidate;
@end

@protocol SBSStatusBarStyleOverridesCoordinatorDelegate;
@protocol SBSStatusBarTapContext;

@interface SBSStatusBarStyleOverridesCoordinator : NSObject
@property (nonatomic, weak, readwrite) id <SBSStatusBarStyleOverridesCoordinatorDelegate> delegate;
@property (nonatomic, readonly) UIStatusBarStyleOverrides styleOverrides;
- (void)setRegisteredStyleOverrides:(UIStatusBarStyleOverrides)styleOverrides reply:(void(^)(NSError *error))reply;
@end

@protocol SBSStatusBarStyleOverridesCoordinatorDelegate <NSObject>
@optional
- (BOOL)statusBarCoordinator:(SBSStatusBarStyleOverridesCoordinator *)coordinator receivedTapWithContext:(id<SBSStatusBarTapContext>)tapContext completionBlock:(void (^)(void))completion;
@required
- (void)statusBarCoordinator:(SBSStatusBarStyleOverridesCoordinator *)coordinator invalidatedRegistrationWithError:(NSError *)error;
@end

@protocol SBSStatusBarTapContext
@property (nonatomic, readonly) UIStatusBarStyleOverrides styleOverride;
@end

#endif // USE(APPLE_INTERNAL_SDK)
