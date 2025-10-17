/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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

#import <UIKit/UIViewController.h>

@class WKWebView;

NS_ASSUME_NONNULL_BEGIN

@protocol WKFullScreenViewControllerDelegate
- (void)requestExitFullScreen;
- (void)showUI;
- (void)hideUI;
@end

@interface WKFullScreenViewController : UIViewController
@property (nonatomic, weak) id <WKFullScreenViewControllerDelegate> delegate;
@property (copy, nonatomic) NSString *location;
@property (assign, nonatomic) BOOL prefersStatusBarHidden;
@property (assign, nonatomic) BOOL prefersHomeIndicatorAutoHidden;
@property (assign, nonatomic, getter=isPlaying) BOOL playing;
@property (assign, nonatomic, getter=isPictureInPictureActive) BOOL pictureInPictureActive;
@property (assign, nonatomic, getter=isinWindowFullscreenActive) BOOL inWindowFullscreenActive;
@property (assign, nonatomic, getter=isAnimating) BOOL animating;

- (id)initWithWebView:(WKWebView *)webView;
- (void)invalidate;
- (void)showUI;
- (void)hideUI;
- (void)showBanner;
- (void)hideBanner;
- (void)videoControlsManagerDidChange;
- (void)setAnimatingViewAlpha:(CGFloat)alpha;
- (void)setSupportedOrientations:(UIInterfaceOrientationMask)supportedOrientations;
- (void)resetSupportedOrientations;
#if ENABLE(VIDEO_USES_ELEMENT_FULLSCREEN)
- (void)hideCustomControls:(BOOL)hidden;
#endif
#if ENABLE(LINEAR_MEDIA_PLAYER)
- (void)configureEnvironmentPickerOrFullscreenVideoButtonView;
#endif
@end

NS_ASSUME_NONNULL_END

#endif // ENABLE(FULLSCREEN_API) && PLATFORM(IOS_FAMILY)
