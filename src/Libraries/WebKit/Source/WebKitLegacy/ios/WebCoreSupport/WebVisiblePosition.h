/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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
#import <Foundation/Foundation.h>
#import <WebKitLegacy/DOMUIKitExtensions.h>

#import <WebKitLegacy/WAKAppKitStubs.h>

typedef struct WebObjectInternal WebObjectInternal;

typedef enum {
    WebTextGranularityCharacter,
    WebTextGranularityWord,
    WebTextGranularitySentence,
    WebTextGranularityParagraph,
    WebTextGranularityLine,
    WebTextGranularityAll,
} WebTextGranularity;

@interface WebVisiblePosition : NSObject {
@private
    WebObjectInternal *_internal;
}

@property (nonatomic) NSSelectionAffinity affinity;

- (NSComparisonResult)compare:(WebVisiblePosition *)other;
- (int)distanceFromPosition:(WebVisiblePosition *)other;
- (WebVisiblePosition *)positionByMovingInDirection:(WebTextAdjustmentDirection)direction amount:(UInt32)amount;
- (WebVisiblePosition *)positionByMovingInDirection:(WebTextAdjustmentDirection)direction amount:(UInt32)amount withAffinityDownstream:(BOOL)affinityDownstream;

// Returnes YES only if a position is at a boundary of a text unit of the specified granularity in the particular direction.
- (BOOL)atBoundaryOfGranularity:(WebTextGranularity)granularity inDirection:(WebTextAdjustmentDirection)direction;

// Returns the next boundary position of a text unit of the given granularity in the given direction, or nil if there is no such position.
- (WebVisiblePosition *)positionOfNextBoundaryOfGranularity:(WebTextGranularity)granularity inDirection:(WebTextAdjustmentDirection)direction;

// Returns YES if position is within a text unit of the given granularity. If the position is at a boundary, returns YES only if
// if the boundary is part of the text unit in the given direction.
- (BOOL)withinTextUnitOfGranularity:(WebTextGranularity)granularity inDirectionIfAtBoundary:(WebTextAdjustmentDirection)direction;

// Returns range of the enclosing text unit of the given granularity, or nil if there is no such enclosing unit. Whether a boundary position
// is enclosed depends on the given direction, using the same rule as -[WebVisiblePosition withinTextUnitOfGranularity:inDirectionAtBoundary:].
- (DOMRange *)enclosingTextUnitOfGranularity:(WebTextGranularity)granularity inDirectionIfAtBoundary:(WebTextAdjustmentDirection)direction;

// Uses fine-tuned logic originally from WebCore::Frame::moveSelectionToStartOrEndOfCurrentWord
- (WebVisiblePosition *)positionAtStartOrEndOfWord;

- (BOOL)isEditable;
- (BOOL)requiresContextForWordBoundary;
- (BOOL)atAlphaNumericBoundaryInDirection:(WebTextAdjustmentDirection)direction;

- (DOMRange *)enclosingRangeWithDictationPhraseAlternatives:(NSArray **)alternatives;
- (DOMRange *)enclosingRangeWithCorrectionIndicator;

@end

@interface DOMRange (VisiblePositionExtensions)

- (WebVisiblePosition *)startPosition;
- (WebVisiblePosition *)endPosition;
+ (DOMRange *)rangeForFirstPosition:(WebVisiblePosition *)first second:(WebVisiblePosition *)second;

// Uses fine-tuned logic from SelectionController::wordSelectionContainingCaretSelection
- (DOMRange *)enclosingWordRange;

@end

@interface DOMNode (VisiblePositionExtensions)

- (WebVisiblePosition *)startPosition;
- (WebVisiblePosition *)endPosition;

@end
