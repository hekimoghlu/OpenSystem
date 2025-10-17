/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#import <WebKit/WKNavigationAction.h>

@class WKNavigation;
@class _WKHitTestResult;
@class _WKUserInitiatedAction;

#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>

typedef NS_ENUM(NSInteger, WKSyntheticClickType) {
    WKSyntheticClickTypeNoTap,
    WKSyntheticClickTypeOneFingerTap,
    WKSyntheticClickTypeTwoFingerTap
} WK_API_AVAILABLE(ios(10.0));
#endif

@interface WKNavigationAction (WKPrivate)

@property (nonatomic, readonly) NSURL *_originalURL;
@property (nonatomic, readonly, getter=_isUserInitiated) BOOL _userInitiated;
@property (nonatomic, readonly) BOOL _canHandleRequest;
@property (nonatomic, readonly) BOOL _shouldOpenExternalSchemes WK_API_AVAILABLE(macos(10.11), ios(9.0));
@property (nonatomic, readonly) BOOL _shouldOpenAppLinks WK_API_AVAILABLE(macos(10.11), ios(9.0));
@property (nonatomic, readonly) BOOL _shouldPerformDownload WK_API_DEPRECATED_WITH_REPLACEMENT("downloadAttribute", macos(10.15, 12.0), ios(13.0, 15.0));
@property (nonatomic, readonly) NSString *_targetFrameName WK_API_AVAILABLE(macos(14.2), ios(17.2));

@property (nonatomic, readonly) BOOL _shouldOpenExternalURLs WK_API_DEPRECATED("use _shouldOpenExternalSchemes and _shouldOpenAppLinks", macos(10.11, 10.11), ios(9.0, 9.0));

@property (nonatomic, readonly) _WKUserInitiatedAction *_userInitiatedAction WK_API_AVAILABLE(macos(10.12), ios(10.0));

#if TARGET_OS_IPHONE
@property (nonatomic, readonly) WKSyntheticClickType _syntheticClickType WK_API_AVAILABLE(ios(10.0));
@property (nonatomic, readonly) CGPoint _clickLocationInRootViewCoordinates WK_API_AVAILABLE(ios(11.0));
#endif

@property (nonatomic, readonly) BOOL _isRedirect WK_API_AVAILABLE(macos(10.13), ios(11.0));
@property (nonatomic, readonly) WKNavigation *_mainFrameNavigation WK_API_AVAILABLE(macos(10.14.4), ios(12.2));
- (void)_storeSKAdNetworkAttribution WK_API_AVAILABLE(macos(13.0), ios(16.1));

@property (nonatomic, readonly) _WKHitTestResult *_hitTestResult WK_API_AVAILABLE(macos(13.3), ios(16.4));

@property (nonatomic, readonly) BOOL _hasOpener WK_API_AVAILABLE(macos(13.3), ios(16.4));

@end
