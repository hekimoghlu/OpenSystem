/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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

#include "FormattingContext.h"
#include "LayoutElementBox.h"
#include "PlacedFloats.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace Layout {

class FloatAvoider;
class Box;

// FloatingContext is responsible for adjusting the position of a box in the current formatting context
// by taking the floating boxes into account.
// Note that a FloatingContext's inline direction always matches the root's inline direction but it may
// not match the PlacedFloats's inline direction (i.e. PlacedFloats may be constructed by a parent BFC with mismatching inline direction).
class FloatingContext {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FloatingContext);
public:
    FloatingContext(const ElementBox& formattingContextRoot, const LayoutState&, const PlacedFloats&);

    const PlacedFloats& placedFloats() const { return m_placedFloats; }

    LayoutPoint positionForFloat(const Box&, const BoxGeometry&, const HorizontalConstraints&) const;
    LayoutPoint positionForNonFloatingFloatAvoider(const Box&, const BoxGeometry&) const;

    struct BlockAxisPositionWithClearance {
        LayoutUnit position;
        std::optional<LayoutUnit> clearance;
    };
    std::optional<BlockAxisPositionWithClearance> blockAxisPositionWithClearance(const Box&, const BoxGeometry&) const;

    bool isEmpty() const { return m_placedFloats.list().isEmpty(); }

    struct Constraints {
        std::optional<PointInContextRoot> start;
        std::optional<PointInContextRoot> end;
    };
    enum class MayBeAboveLastFloat : bool { No, Yes };
    Constraints constraints(LayoutUnit candidateTop, LayoutUnit candidateBottom, MayBeAboveLastFloat) const;

    PlacedFloats::Item makeFloatItem(const Box& floatBox, const BoxGeometry&, std::optional<size_t> line = { }) const;

    bool isStartPositioned(const Box& floatBox) const;

private:
    bool isFloatingCandidateStartPositionedInBlockFormattingContext(const Box&) const;
    Clear clearInBlockFormattingContext(const Box&) const;

    const ElementBox& root() const { return m_formattingContextRoot; }
    // FIXME: Turn this into an actual geometry cache.
    const LayoutState& containingBlockGeometries() const { return m_layoutState; }

    void findPositionForFormattingContextRoot(FloatAvoider&, BoxGeometry::HorizontalEdges containingBlockContentBoxEdges) const;

    struct AbsoluteCoordinateValuesForFloatAvoider;
    AbsoluteCoordinateValuesForFloatAvoider absoluteCoordinates(const Box&, LayoutPoint borderBoxTopLeft) const;
    LayoutPoint mapTopLeftToBlockFormattingContextRoot(const Box&, LayoutPoint borderBoxTopLeft) const;
    Point mapPointFromFloatingContextRootToBlockFormattingContextRoot(Point) const;

    CheckedRef<const ElementBox> m_formattingContextRoot;
    const LayoutState& m_layoutState;
    const PlacedFloats& m_placedFloats;
};

}
}
