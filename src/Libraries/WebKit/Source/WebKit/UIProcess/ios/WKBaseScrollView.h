/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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

#if PLATFORM(IOS_FAMILY)

#import "WKBrowserEngineDefinitions.h"
#import <UIKit/UIKit.h>

#if ENABLE(OVERLAY_REGIONS_IN_EVENT_REGION)
#import <WebCore/IntRectHash.h>
#import <wtf/HashSet.h>
#endif

@class WKBEScrollViewScrollUpdate;
@class WKBaseScrollView;

@protocol WKBaseScrollViewDelegate <NSObject>

- (BOOL)shouldAllowPanGestureRecognizerToReceiveTouchesInScrollView:(WKBaseScrollView *)scrollView;
- (UIAxis)axesToPreventScrollingForPanGestureInScrollView:(WKBaseScrollView *)scrollView;

@end

@interface WKBaseScrollView : WKBEScrollView

@property (nonatomic, weak) id<WKBaseScrollViewDelegate> baseScrollViewDelegate;
@property (nonatomic, readonly) UIAxis axesToPreventMomentumScrolling;

#if ENABLE(OVERLAY_REGIONS_IN_EVENT_REGION)
@property (nonatomic) NSUInteger _scrollingBehavior;
@property (nonatomic, readonly, getter=overlayRegionsForTesting) HashSet<WebCore::IntRect>& overlayRegionRects;

- (BOOL)_hasEnoughContentForOverlayRegions;
- (void)_updateOverlayRegionRects:(const HashSet<WebCore::IntRect>&)overlayRegions;
- (void)_updateOverlayRegionsBehavior:(BOOL)selected;
#endif // ENABLE(OVERLAY_REGIONS_IN_EVENT_REGION)

@end

#endif // PLATFORM(IOS_FAMILY)
