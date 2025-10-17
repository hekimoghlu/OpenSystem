/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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
#import <WebKitLegacy/WebFrameView.h>
#if TARGET_OS_IPHONE
#import <WebKitLegacy/WAKAppKitStubs.h>
#endif

@interface WebFrameView (WebPrivate)

// FIXME: This method was used by Safari 4.0.x and older versions, but has not been used by any other WebKit
// clients to my knowledge, and will not be used by future versions of Safari. It can probably be removed 
// once we no longer need to keep nightly WebKit builds working with Safari 4.0.x and earlier.
/*!
    @method _largestChildWithScrollBars
    @abstract Of the child WebFrameViews that are displaying scroll bars, determines which has the largest area.
    @result A child WebFrameView that is displaying scroll bars, or nil if none.
 */
- (WebFrameView *)_largestChildWithScrollBars;

// FIXME: This method was used by Safari 4.0.x and older versions, but has not been used by any other WebKit
// clients to my knowledge, and will not be used by future versions of Safari. It can probably be removed 
// once we no longer need to keep nightly WebKit builds working with Safari 4.0.x and earlier.
/*!
    @method _hasScrollBars
    @result YES if at least one scroll bar is currently displayed
 */
- (BOOL)_hasScrollBars;

/*!
    @method _largestScrollableChild
    @abstract Of the child WebFrameViews that allow scrolling, determines which has the largest area.
    @result A child WebFrameView that is scrollable, or nil if none.
 */
- (WebFrameView *)_largestScrollableChild;

/*!
    @method _isScrollable
    @result YES if scrolling is currently possible, whether or not scroll bars are currently showing. This
    differs from -allowsScrolling in that the latter method only checks whether scrolling has been
    explicitly disallowed via a call to setAllowsScrolling:NO.
 */
- (BOOL)_isScrollable;

/*!
    @method _contentView
    @result The content view (NSClipView) of the WebFrameView's scroll view.
 */
#if TARGET_OS_IPHONE
- (WAKClipView *)_contentView;
#else
- (NSClipView *)_contentView;
#endif

/*!
    @method _customScrollViewClass
    @result The custom scroll view class that is installed, nil if the default scroll view is being used.
 */
- (Class)_customScrollViewClass;

#if !TARGET_OS_IPHONE
/*!
    @method _setCustomScrollViewClass:
    @abstract Switches the WebFrameView's scroll view class, this class needs to be a subclass of WebDynamicScrollBarsView.
    Passing nil will switch back to the default WebDynamicScrollBarsView class.
 */
- (void)_setCustomScrollViewClass:(Class)scrollViewClass;
#endif

@end
