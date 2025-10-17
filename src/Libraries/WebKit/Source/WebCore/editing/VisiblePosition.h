/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
#include "Position.h"

namespace WebCore {

class VisiblePosition {
public:
    // VisiblePosition default affinity is downstream for callers that do not really care because it is more efficient than upstream.
    static constexpr auto defaultAffinity = Affinity::Downstream;

    VisiblePosition() = default;

    // This constructor will ignore the passed-in affinity if the position is not at the end of a line.
    WEBCORE_EXPORT VisiblePosition(const Position&, Affinity = defaultAffinity);

    bool isNull() const { return m_deepPosition.isNull(); }
    bool isNotNull() const { return m_deepPosition.isNotNull(); }
    bool isOrphan() const { return m_deepPosition.isOrphan(); }

    Position deepEquivalent() const { return m_deepPosition; }
    Affinity affinity() const { return m_affinity; }

    void setAffinity(Affinity affinity) { m_affinity = affinity; }

    // FIXME: Change the following functions' parameter from a boolean to StayInEditableContent.

    // next() and previous() increment/decrement by a character cluster.
    WEBCORE_EXPORT VisiblePosition next(EditingBoundaryCrossingRule = CanCrossEditingBoundary, bool* reachedBoundary = nullptr) const;
    WEBCORE_EXPORT VisiblePosition previous(EditingBoundaryCrossingRule = CanCrossEditingBoundary, bool* reachedBoundary = nullptr) const;

    VisiblePosition honorEditingBoundaryAtOrBefore(const VisiblePosition&, bool* reachedBoundary = nullptr) const;
    VisiblePosition honorEditingBoundaryAtOrAfter(const VisiblePosition&, bool* reachedBoundary = nullptr) const;

    WEBCORE_EXPORT VisiblePosition left(bool stayInEditableContent = false, bool* reachedBoundary = nullptr) const;
    WEBCORE_EXPORT VisiblePosition right(bool stayInEditableContent = false, bool* reachedBoundary = nullptr) const;

    WEBCORE_EXPORT char32_t characterAfter() const;
    char32_t characterBefore() const { return previous().characterAfter(); }

    // FIXME: This does not handle [table, 0] correctly.
    Element* rootEditableElement() const { return m_deepPosition.isNotNull() ? m_deepPosition.deprecatedNode()->rootEditableElement() : 0; }

    InlineBoxAndOffset inlineBoxAndOffset() const;
    InlineBoxAndOffset inlineBoxAndOffset(TextDirection primaryDirection) const;

    struct LocalCaretRect {
        LayoutRect rect;
        RenderObject* renderer { nullptr };
    };
    WEBCORE_EXPORT LocalCaretRect localCaretRect() const;

    // Bounds of (possibly transformed) caret in absolute coords.
    WEBCORE_EXPORT IntRect absoluteCaretBounds(bool* insideFixed = nullptr) const;

    // Abs x/y position of the caret ignoring transforms.
    // FIXME: navigation with transforms should be smarter.
    WEBCORE_EXPORT int lineDirectionPointForBlockDirectionNavigation() const;

    WEBCORE_EXPORT FloatRect absoluteSelectionBoundsForLine() const;

    // This is a tentative enhancement of operator== to account for affinity.
    // FIXME: Combine this function with operator==.
    bool equals(const VisiblePosition&) const;

#if ENABLE(TREE_DEBUGGING)
    void debugPosition(ASCIILiteral msg = ""_s) const;
    String debugDescription() const;
    void showTreeForThis() const;
#endif

private:
    static Position canonicalPosition(const Position&);

    Position leftVisuallyDistinctCandidate() const;
    Position rightVisuallyDistinctCandidate() const;

    Position m_deepPosition;
    Affinity m_affinity { defaultAffinity };
};

bool operator==(const VisiblePosition&, const VisiblePosition&);

WEBCORE_EXPORT std::partial_ordering documentOrder(const VisiblePosition&, const VisiblePosition&);
bool operator<(const VisiblePosition&, const VisiblePosition&);
bool operator>(const VisiblePosition&, const VisiblePosition&);
bool operator<=(const VisiblePosition&, const VisiblePosition&);
bool operator>=(const VisiblePosition&, const VisiblePosition&);

WEBCORE_EXPORT std::optional<BoundaryPoint> makeBoundaryPoint(const VisiblePosition&);

WEBCORE_EXPORT Element* enclosingBlockFlowElement(const VisiblePosition&);

bool isFirstVisiblePositionInNode(const VisiblePosition&, const Node*);
bool isLastVisiblePositionInNode(const VisiblePosition&, const Node*);

bool areVisiblePositionsInSameTreeScope(const VisiblePosition&, const VisiblePosition&);

Node* commonInclusiveAncestor(const VisiblePosition&, const VisiblePosition&);

WTF::TextStream& operator<<(WTF::TextStream&, Affinity);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const VisiblePosition&);

struct VisiblePositionRange {
    VisiblePosition start;
    VisiblePosition end;

    bool isNull() const { return start.isNull() || end.isNull(); }
#if ENABLE(TREE_DEBUGGING)
    String debugDescription() const;
#endif
};

WEBCORE_EXPORT std::optional<SimpleRange> makeSimpleRange(const VisiblePositionRange&);
WEBCORE_EXPORT VisiblePositionRange makeVisiblePositionRange(const std::optional<SimpleRange>&);

Node* commonInclusiveAncestor(const VisiblePositionRange&);

WEBCORE_EXPORT bool intersects(const VisiblePositionRange&, const VisiblePositionRange&);
WEBCORE_EXPORT bool contains(const VisiblePositionRange&, const VisiblePosition&);
WEBCORE_EXPORT VisiblePositionRange intersection(const VisiblePositionRange&, const VisiblePositionRange&);
WEBCORE_EXPORT VisiblePosition midpoint(const VisiblePositionRange&);

// inlines

inline bool operator==(const VisiblePosition& a, const VisiblePosition& b)
{
    // FIXME: Is it correct and helpful for this to be ignoring differences in affinity?
    return a.deepEquivalent() == b.deepEquivalent();
}

inline bool operator<(const VisiblePosition& a, const VisiblePosition& b)
{
    return is_lt(documentOrder(a, b));
}

inline bool operator>(const VisiblePosition& a, const VisiblePosition& b)
{
    return is_gt(documentOrder(a, b));
}

inline bool operator<=(const VisiblePosition& a, const VisiblePosition& b)
{
    return is_lteq(documentOrder(a, b));
}

inline bool operator>=(const VisiblePosition& a, const VisiblePosition& b)
{
    return is_gteq(documentOrder(a, b));
}

} // namespace WebCore

#if ENABLE(TREE_DEBUGGING)
// Outside the WebCore namespace for ease of invocation from the debugger.
void showTree(const WebCore::VisiblePosition*);
void showTree(const WebCore::VisiblePosition&);
#endif
