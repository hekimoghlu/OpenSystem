/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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

#include "EditingBoundary.h"
#include "VisibleSelection.h"

namespace WebCore {

class Node;
class LayoutUnit;
class VisiblePosition;
class SimplifiedBackwardsTextIterator;
class TextIterator;

enum class WordSide : bool { RightWordIfOnBoundary, LeftWordIfOnBoundary };

// words
WEBCORE_EXPORT VisiblePosition startOfWord(const VisiblePosition&, WordSide = WordSide::RightWordIfOnBoundary);
WEBCORE_EXPORT VisiblePosition endOfWord(const VisiblePosition&, WordSide = WordSide::RightWordIfOnBoundary);
WEBCORE_EXPORT VisiblePosition previousWordPosition(const VisiblePosition&);
WEBCORE_EXPORT VisiblePosition nextWordPosition(const VisiblePosition&);
WEBCORE_EXPORT VisiblePosition rightWordPosition(const VisiblePosition&, bool skipsSpaceWhenMovingRight);
WEBCORE_EXPORT VisiblePosition leftWordPosition(const VisiblePosition&, bool skipsSpaceWhenMovingRight);
bool isStartOfWord(const VisiblePosition&);

// sentences
WEBCORE_EXPORT VisiblePosition startOfSentence(const VisiblePosition&);
WEBCORE_EXPORT VisiblePosition endOfSentence(const VisiblePosition&);
WEBCORE_EXPORT VisiblePosition previousSentencePosition(const VisiblePosition&);
WEBCORE_EXPORT VisiblePosition nextSentencePosition(const VisiblePosition&);

// lines
WEBCORE_EXPORT VisiblePosition startOfLine(const VisiblePosition&);
WEBCORE_EXPORT VisiblePosition endOfLine(const VisiblePosition&);
WEBCORE_EXPORT VisiblePosition previousLinePosition(const VisiblePosition&, LayoutUnit lineDirectionPoint, EditableType = ContentIsEditable);
WEBCORE_EXPORT VisiblePosition nextLinePosition(const VisiblePosition&, LayoutUnit lineDirectionPoint, EditableType = ContentIsEditable);
WEBCORE_EXPORT bool inSameLine(const VisiblePosition&, const VisiblePosition&);
WEBCORE_EXPORT bool isStartOfLine(const VisiblePosition&);
WEBCORE_EXPORT bool isEndOfLine(const VisiblePosition&);
VisiblePosition logicalStartOfLine(const VisiblePosition&, bool* reachedBoundary = nullptr);
VisiblePosition logicalEndOfLine(const VisiblePosition&, bool* reachedBoundary = nullptr);
bool isLogicalEndOfLine(const VisiblePosition&);
VisiblePosition leftBoundaryOfLine(const VisiblePosition&, TextDirection, bool* reachedBoundary);
VisiblePosition rightBoundaryOfLine(const VisiblePosition&, TextDirection, bool* reachedBoundary);

// paragraphs (perhaps a misnomer, can be divided by line break elements)
WEBCORE_EXPORT VisiblePosition startOfParagraph(const VisiblePosition&, EditingBoundaryCrossingRule = CannotCrossEditingBoundary);
WEBCORE_EXPORT VisiblePosition endOfParagraph(const VisiblePosition&, EditingBoundaryCrossingRule = CannotCrossEditingBoundary);
VisiblePosition startOfNextParagraph(const VisiblePosition&);
WEBCORE_EXPORT VisiblePosition previousParagraphPosition(const VisiblePosition&, LayoutUnit x);
WEBCORE_EXPORT VisiblePosition nextParagraphPosition(const VisiblePosition&, LayoutUnit x);
WEBCORE_EXPORT bool isStartOfParagraph(const VisiblePosition&, EditingBoundaryCrossingRule = CannotCrossEditingBoundary);
WEBCORE_EXPORT bool isEndOfParagraph(const VisiblePosition&, EditingBoundaryCrossingRule = CannotCrossEditingBoundary);
bool inSameParagraph(const VisiblePosition&, const VisiblePosition&, EditingBoundaryCrossingRule = CannotCrossEditingBoundary);
bool isBlankParagraph(const VisiblePosition&);

// blocks (true paragraphs; line break elements don't break blocks)
VisiblePosition startOfBlock(const VisiblePosition&, EditingBoundaryCrossingRule = CannotCrossEditingBoundary);
VisiblePosition endOfBlock(const VisiblePosition&, EditingBoundaryCrossingRule = CannotCrossEditingBoundary);
bool inSameBlock(const VisiblePosition&, const VisiblePosition&);
bool isStartOfBlock(const VisiblePosition&);
bool isEndOfBlock(const VisiblePosition&);

// document
WEBCORE_EXPORT VisiblePosition startOfDocument(const Node*);
WEBCORE_EXPORT VisiblePosition endOfDocument(const Node*);
WEBCORE_EXPORT VisiblePosition startOfDocument(const VisiblePosition&);
WEBCORE_EXPORT VisiblePosition endOfDocument(const VisiblePosition&);
WEBCORE_EXPORT bool isStartOfDocument(const VisiblePosition&);
WEBCORE_EXPORT bool isEndOfDocument(const VisiblePosition&);

// editable content
WEBCORE_EXPORT VisiblePosition startOfEditableContent(const VisiblePosition&);
WEBCORE_EXPORT VisiblePosition endOfEditableContent(const VisiblePosition&);
WEBCORE_EXPORT bool isEndOfEditableOrNonEditableContent(const VisiblePosition&);

WEBCORE_EXPORT bool atBoundaryOfGranularity(const VisiblePosition&, TextGranularity, SelectionDirection);
WEBCORE_EXPORT bool withinTextUnitOfGranularity(const VisiblePosition&, TextGranularity, SelectionDirection);
WEBCORE_EXPORT VisiblePosition positionOfNextBoundaryOfGranularity(const VisiblePosition&, TextGranularity, SelectionDirection);
WEBCORE_EXPORT std::optional<SimpleRange> enclosingTextUnitOfGranularity(const VisiblePosition&, TextGranularity, SelectionDirection);
WEBCORE_EXPORT std::ptrdiff_t distanceBetweenPositions(const VisiblePosition&, const VisiblePosition&);
WEBCORE_EXPORT std::optional<SimpleRange> wordRangeFromPosition(const VisiblePosition&);
WEBCORE_EXPORT VisiblePosition closestWordBoundaryForPosition(const VisiblePosition& position);
WEBCORE_EXPORT void charactersAroundPosition(const VisiblePosition&, char32_t& oneAfter, char32_t& oneBefore, char32_t& twoBefore);
WEBCORE_EXPORT std::optional<SimpleRange> rangeExpandedAroundPositionByCharacters(const VisiblePosition&, int numberOfCharactersToExpand);
WEBCORE_EXPORT std::optional<SimpleRange> rangeExpandedAroundRangeByCharacters(const VisibleSelection&, uint64_t numberOfCharactersToExpandBackwards, uint64_t numberOfCharactersToExpandForwards);
WEBCORE_EXPORT std::optional<SimpleRange> rangeExpandedByCharactersInDirectionAtWordBoundary(const VisiblePosition&, int numberOfCharactersToExpand, SelectionDirection);
enum class WithinWordBoundary : bool { No, Yes };
WEBCORE_EXPORT std::pair<VisiblePosition, WithinWordBoundary> wordBoundaryForPositionWithoutCrossingLine(const VisiblePosition&);

// helper function
enum BoundarySearchContextAvailability { DontHaveMoreContext, MayHaveMoreContext };
typedef unsigned (*BoundarySearchFunction)(StringView, unsigned offset, BoundarySearchContextAvailability, bool& needMoreContext);
unsigned startWordBoundary(StringView, unsigned, BoundarySearchContextAvailability, bool&);
unsigned endWordBoundary(StringView, unsigned, BoundarySearchContextAvailability, bool&);
unsigned startSentenceBoundary(StringView, unsigned, BoundarySearchContextAvailability, bool&);
unsigned endSentenceBoundary(StringView, unsigned, BoundarySearchContextAvailability, bool&);
unsigned suffixLengthForRange(const SimpleRange&, Vector<UChar, 1024>&);
unsigned prefixLengthForRange(const SimpleRange&, Vector<UChar, 1024>&);
unsigned backwardSearchForBoundaryWithTextIterator(SimplifiedBackwardsTextIterator&, Vector<UChar, 1024>&, unsigned, BoundarySearchFunction);
unsigned forwardSearchForBoundaryWithTextIterator(TextIterator&, Vector<UChar, 1024>&, unsigned, BoundarySearchFunction);
RefPtr<Node> findStartOfParagraph(Node*, Node*, Node*, int&, Position::AnchorType&, EditingBoundaryCrossingRule);
RefPtr<Node> findEndOfParagraph(Node*, Node*, Node*, int&, Position::AnchorType&, EditingBoundaryCrossingRule);

} // namespace WebCore
