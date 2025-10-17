/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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
#if !PLATFORM(IOS_FAMILY)

#import "WebDynamicScrollBarsView.h"
#import <WebCore/WebCoreFrameView.h>

@interface WebDynamicScrollBarsView (WebInternal) <WebCoreFrameScrollView>

- (BOOL)allowsHorizontalScrolling;
- (BOOL)allowsVerticalScrolling;

- (void)setScrollingModes:(WebCore::ScrollbarMode)hMode vertical:(WebCore::ScrollbarMode)vMode andLock:(BOOL)lock;
- (void)scrollingModes:(WebCore::ScrollbarMode*)hMode vertical:(WebCore::ScrollbarMode*)vMode;

- (WebCore::ScrollbarMode)horizontalScrollingMode;
- (WebCore::ScrollbarMode)verticalScrollingMode;

- (void)setHorizontalScrollingMode:(WebCore::ScrollbarMode)mode andLock:(BOOL)lock;
- (void)setVerticalScrollingMode:(WebCore::ScrollbarMode)mode andLock:(BOOL)lock;

- (void)setHorizontalScrollingModeLocked:(BOOL)locked;
- (void)setVerticalScrollingModeLocked:(BOOL)locked;
- (void)setScrollingModesLocked:(BOOL)mode;

- (BOOL)horizontalScrollingModeLocked;
- (BOOL)verticalScrollingModeLocked;

- (void)updateScrollers;
- (void)setSuppressLayout:(BOOL)flag;

// Calculate the appropriate frame for the contentView based on allowsScrollersToOverlapContent.
- (NSRect)contentViewFrame;

// Returns YES if we're currently in the middle of programmatically moving the
// scrollbar.
// NOTE: As opposed to other places in the code, programmatically moving the
// scrollers from inside this class should not fire JS events.
- (BOOL)inProgrammaticScroll;
@end

#endif // !PLATFORM(IOS_FAMILY)
