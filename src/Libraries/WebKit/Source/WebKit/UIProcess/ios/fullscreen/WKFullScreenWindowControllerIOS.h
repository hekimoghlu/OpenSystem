/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 7, 2022.
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
#if ENABLE(FULLSCREEN_API) && PLATFORM(IOS_FAMILY)

#import <UIKit/UIViewControllerTransitioning.h>
#import <wtf/CompletionHandler.h>

@class WKFullScreenViewController;
@class WKWebView;

@interface WKFullScreenWindowController : NSObject <UIViewControllerTransitioningDelegate>
@property (readonly, retain, nonatomic) UIView *webViewPlaceholder;
@property (readonly, retain, nonatomic) WKFullScreenViewController *fullScreenViewController;
@property (readonly, assign, nonatomic) BOOL isFullScreen;
#if PLATFORM(VISION)
@property (readonly, assign, nonatomic) BOOL prefersSceneDimming;
#endif
#if ENABLE(QUICKLOOK_FULLSCREEN)
@property (readonly, assign, nonatomic) BOOL isUsingQuickLook;
@property (readonly, assign, nonatomic) CGSize imageDimensions;
#endif

- (id)initWithWebView:(WKWebView *)webView;
- (void)enterFullScreen:(CGSize)mediaDimensions completionHandler:(CompletionHandler<void(bool)>&&)completionHandler;
#if ENABLE(QUICKLOOK_FULLSCREEN)
- (void)updateImageSource;
#endif
- (void)beganEnterFullScreenWithInitialFrame:(CGRect)initialFrame finalFrame:(CGRect)finalFrame;
- (void)requestRestoreFullScreen:(CompletionHandler<void(bool)>&&)completionHandler;
- (void)requestExitFullScreen;
- (void)exitFullScreen;
- (void)beganExitFullScreenWithInitialFrame:(CGRect)initialFrame finalFrame:(CGRect)finalFrame;
- (void)setSupportedOrientations:(UIInterfaceOrientationMask)orientations;
- (void)resetSupportedOrientations;
- (void)close;
- (void)webViewDidRemoveFromSuperviewWhileInFullscreen;
- (void)videoControlsManagerDidChange;
- (void)didCleanupFullscreen;

#if PLATFORM(VISION)
- (void)toggleSceneDimming;
#endif

@end

#endif
