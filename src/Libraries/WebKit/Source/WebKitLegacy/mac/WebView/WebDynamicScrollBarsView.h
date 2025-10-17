/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#if TARGET_OS_IPHONE
// In iOS WebKit, WebDynamicScrollBarsView is a WAKScrollView.
// See WebCore/WAKAppKitStubs.h.
#else
// This is a Private header (containing SPI), despite the fact that its name
// does not contain the word Private.

#import <AppKit/NSScrollView.h>

struct WebDynamicScrollBarsViewPrivate;
@interface WebDynamicScrollBarsView : NSScrollView {
@private
    struct WebDynamicScrollBarsViewPrivate *_private;

#ifndef __OBJC2__
    // We need to pad the class out to its former size.  See <rdar://problem/7814899> for more information.
    char padding[16];
#endif
}

// For use by DumpRenderTree only.
+ (void)setCustomScrollerClass:(Class)scrollerClass;

// This was originally added for Safari's benefit, but Safari has not used it for a long time.
// Perhaps it can be removed.
- (void)setAllowsHorizontalScrolling:(BOOL)flag;

// Determines whether the scrollers should be drawn outside of the content (as in normal scroll views)
// or should overlap the content.
- (void)setAllowsScrollersToOverlapContent:(BOOL)flag;

// These methods hide the scrollers in a way that does not prevent scrolling.
- (void)setAlwaysHideHorizontalScroller:(BOOL)flag;
- (void)setAlwaysHideVerticalScroller:(BOOL)flag;

// These methods return YES if the scrollers are visible, or if the only reason that they are not
// visible is that they have been suppressed by setAlwaysHideHorizontal/VerticalScroller:.
- (BOOL)horizontalScrollingAllowed;
- (BOOL)verticalScrollingAllowed;
@end

#endif // !TARGET_OS_IPHONE
