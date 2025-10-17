/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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

#include "FocusDirection.h"
#include "HTMLFrameOwnerElement.h"
#include "LayoutRect.h"
#include "Node.h"
#include <limits>

namespace WebCore {

class Element;
class HTMLAreaElement;
class IntRect;
class LocalFrame;
class RenderObject;

inline long long maxDistance()
{
    return std::numeric_limits<long long>::max();
}

inline int fudgeFactor()
{
    return 2;
}

bool isSpatialNavigationEnabled(const LocalFrame*);

// Spatially speaking, two given elements in a web page can be:
// 1) Fully aligned: There is a full intersection between the rects, either
//    vertically or horizontally.
//
// * Horizontally       * Vertically
//    _
//   |_|                   _ _ _ _ _ _
//   |_|...... _          |_|_|_|_|_|_|
//   |_|      |_|         .       .
//   |_|......|_|   OR    .       .
//   |_|      |_|         .       .
//   |_|......|_|          _ _ _ _
//   |_|                  |_|_|_|_|
//
//
// 2) Partially aligned: There is a partial intersection between the rects, either
//    vertically or horizontally.
//
// * Horizontally       * Vertically
//    _                   _ _ _ _ _
//   |_|                 |_|_|_|_|_|
//   |_|.... _      OR           . .
//   |_|    |_|                  . .
//   |_|....|_|                  ._._ _
//          |_|                  |_|_|_|
//          |_|
//
// 3) Or, otherwise, not aligned at all.
//
// * Horizontally       * Vertically
//         _              _ _ _ _
//        |_|            |_|_|_|_|
//        |_|                    .
//        |_|                     .
//       .          OR             .
//    _ .                           ._ _ _ _ _
//   |_|                            |_|_|_|_|_|
//   |_|
//   |_|
//
// "Totally Aligned" elements are preferable candidates to move
// focus to over "Partially Aligned" ones, that on its turns are
// more preferable than "Not Aligned".
enum class RectsAlignment {
    None = 0,
    Partial,
    Full
};

struct FocusCandidate {
    FocusCandidate()
        : visibleNode(nullptr)
        , focusableNode(nullptr)
        , enclosingScrollableBox(nullptr)
        , distance(maxDistance())
        , alignment(RectsAlignment::None)
        , isOffscreen(true)
        , isOffscreenAfterScrolling(true)
    {
    }

    FocusCandidate(Node* n, FocusDirection);
    explicit FocusCandidate(HTMLAreaElement* area, FocusDirection);
    bool isNull() const { return !visibleNode; }
    bool inScrollableContainer() const { return visibleNode && enclosingScrollableBox; }
    Document* document() const { return visibleNode ? &visibleNode->document() : 0; }

    // We handle differently visibleNode and FocusableNode to properly handle the areas of imagemaps,
    // where visibleNode would represent the image element and focusableNode would represent the area element.
    // In all other cases, visibleNode and focusableNode are one and the same.
    WeakPtr<Node, WeakPtrImplWithEventTargetData> visibleNode;
    WeakPtr<Node, WeakPtrImplWithEventTargetData> focusableNode;
    WeakPtr<Node, WeakPtrImplWithEventTargetData> enclosingScrollableBox;
    long long distance;
    RectsAlignment alignment;
    LayoutRect rect;
    bool isOffscreen;
    bool isOffscreenAfterScrolling;
};

bool hasOffscreenRect(Node*, FocusDirection = FocusDirection::None);
bool scrollInDirection(LocalFrame*, FocusDirection);
bool scrollInDirection(Node* container, FocusDirection);
bool canScrollInDirection(const Node* container, FocusDirection);
bool canScrollInDirection(const LocalFrame*, FocusDirection);
bool canBeScrolledIntoView(FocusDirection, const FocusCandidate&);
bool areElementsOnSameLine(const FocusCandidate& firstCandidate, const FocusCandidate& secondCandidate);
bool isValidCandidate(FocusDirection, const FocusCandidate&, FocusCandidate&);
void distanceDataForNode(FocusDirection, const FocusCandidate& current, FocusCandidate& candidate);
Node* scrollableEnclosingBoxOrParentFrameForNodeInDirection(FocusDirection, Node*);
LayoutRect nodeRectInAbsoluteCoordinates(Node*, bool ignoreBorder = false);
LayoutRect frameRectInAbsoluteCoordinates(LocalFrame*);
LayoutRect virtualRectForDirection(FocusDirection, const LayoutRect& startingRect, LayoutUnit width = 0_lu);
LayoutRect virtualRectForAreaElementAndDirection(HTMLAreaElement*, FocusDirection);
HTMLFrameOwnerElement* frameOwnerElement(FocusCandidate&);

} // namespace WebCore
