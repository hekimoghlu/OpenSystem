/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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

#import <CoreGraphics/CoreGraphics.h>
#import <WebKitLegacy/WebFrame.h>
#import <WebKitLegacy/WebVisiblePosition.h>

@class DOMRange;
@class DOMVisiblePosition;

typedef enum {
    WebTextSelectionStateNone,
    WebTextSelectionStateCaret,
    WebTextSelectionStateRange,
} WebTextSelectionState;

typedef enum {
    WebTextSmartExtendDirectionNone,
    WebTextSmartExtendDirectionLeft,
    WebTextSmartExtendDirectionRight,
} WebTextSmartExtendDirection;

@interface WebFrame (WebFrameIOS)

- (void)moveSelectionToPoint:(CGPoint)point;

- (void)clearSelection;
- (BOOL)hasSelection;
- (WebTextSelectionState)selectionState;
- (CGRect)caretRectForPosition:(WebVisiblePosition *)position;
- (CGRect)closestCaretRectInMarkedTextRangeForPoint:(CGPoint)point;
- (void)collapseSelection;
- (NSArray *)selectionRects;
- (NSArray *)selectionRectsForRange:(DOMRange *)domRange;
- (DOMRange *)wordAtPoint:(CGPoint)point;
- (WebVisiblePosition *)webVisiblePositionForPoint:(CGPoint)point;
- (void)setRangedSelectionBaseToCurrentSelection;
- (void)setRangedSelectionBaseToCurrentSelectionStart;
- (void)setRangedSelectionBaseToCurrentSelectionEnd;
- (void)clearRangedSelectionInitialExtent;
- (void)setRangedSelectionInitialExtentToCurrentSelectionStart;
- (void)setRangedSelectionInitialExtentToCurrentSelectionEnd;
- (BOOL)setRangedSelectionExtentPoint:(CGPoint)extentPoint baseIsStart:(BOOL)baseIsStart allowFlipping:(BOOL)allowFlipping;
- (BOOL)setSelectionWithBasePoint:(CGPoint)basePoint extentPoint:(CGPoint)extentPoint baseIsStart:(BOOL)baseIsStart;
- (BOOL)setSelectionWithBasePoint:(CGPoint)basePoint extentPoint:(CGPoint)extentPoint baseIsStart:(BOOL)baseIsStart allowFlipping:(BOOL)allowFlipping;
- (void)setSelectionWithFirstPoint:(CGPoint)firstPoint secondPoint:(CGPoint)secondPoint;
- (void)ensureRangedSelectionContainsInitialStartPoint:(CGPoint)initialStartPoint initialEndPoint:(CGPoint)initialEndPoint;

- (void)smartExtendRangedSelection:(WebTextSmartExtendDirection)direction;
- (void)aggressivelyExpandSelectionToWordContainingCaretSelection; // Doesn't accept no for an answer; expands past white space.

- (WKWritingDirection)selectionBaseWritingDirection;
- (void)toggleBaseWritingDirection;
- (void)setBaseWritingDirection:(WKWritingDirection)direction;

- (void)moveSelectionToStart;
- (void)moveSelectionToEnd;

- (void)setSelectionGranularity:(WebTextGranularity)granularity;
- (void)setRangedSelectionWithExtentPoint:(CGPoint)point;

- (WebVisiblePosition *)startPosition;
- (WebVisiblePosition *)endPosition;

- (BOOL)renderedCharactersExceed:(NSUInteger)threshold;

- (CGRect)elementRectAtPoint:(CGPoint)point;
@end

#endif // TARGET_OS_IPHONE
