/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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

#include "ElementIdentifier.h"
#include "FloatRect.h"
#include "LayoutRect.h"
#include "LayoutUnit.h"
#include "ScrollTypes.h"
#include "StyleScrollSnapPoints.h"
#include <utility>
#include <wtf/Vector.h>

namespace WebCore {

class ScrollableArea;
class RenderBox;
class RenderStyle;
class Element;

template <typename T>
struct SnapOffset {
    T offset;
    ScrollSnapStop stop;
    bool hasSnapAreaLargerThanViewport;
    Markable<ElementIdentifier> snapTargetID;
    bool isFocused;
    Vector<size_t> snapAreaIndices;
};

template <typename UnitType, typename RectType>
struct ScrollSnapOffsetsInfo {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    ScrollSnapStrictness strictness { ScrollSnapStrictness::None };
    Vector<SnapOffset<UnitType>> horizontalSnapOffsets;
    Vector<SnapOffset<UnitType>> verticalSnapOffsets;
    Vector<RectType> snapAreas;
    Vector<ElementIdentifier> snapAreasIDs;

    bool isEqual(const ScrollSnapOffsetsInfo<UnitType, RectType>& other) const
    {
        return strictness == other.strictness && horizontalSnapOffsets == other.horizontalSnapOffsets && verticalSnapOffsets == other.verticalSnapOffsets && snapAreas == other.snapAreas;
    }

    bool isEmpty() const
    {
        return horizontalSnapOffsets.isEmpty() && verticalSnapOffsets.isEmpty();
    }

    Vector<SnapOffset<UnitType>> offsetsForAxis(ScrollEventAxis axis) const
    {
        return axis == ScrollEventAxis::Vertical ? verticalSnapOffsets : horizontalSnapOffsets;
    }

    template<typename OutputType> OutputType convertUnits(float deviceScaleFactor = 0.0) const;
    template<typename SizeType, typename PointType>
    WEBCORE_EXPORT std::pair<UnitType, std::optional<unsigned>> closestSnapOffset(ScrollEventAxis, const SizeType& viewportSize, PointType scrollDestinationOffset, float velocity, std::optional<UnitType> originalPositionForDirectionalSnapping = std::nullopt) const;
};

template<typename UnitType> inline bool operator==(const SnapOffset<UnitType>& a, const SnapOffset<UnitType>& b)
{
    return a.offset == b.offset && a.stop == b.stop && a.snapAreaIndices == b.snapAreaIndices;
}

using LayoutScrollSnapOffsetsInfo = ScrollSnapOffsetsInfo<LayoutUnit, LayoutRect>;
using FloatScrollSnapOffsetsInfo = ScrollSnapOffsetsInfo<float, FloatRect>;
using FloatSnapOffset = SnapOffset<float>;

template <> template <>
LayoutScrollSnapOffsetsInfo FloatScrollSnapOffsetsInfo::convertUnits(float /* unusedScaleFactor */) const;
template <> template <>
WEBCORE_EXPORT std::pair<float, std::optional<unsigned>> FloatScrollSnapOffsetsInfo::closestSnapOffset(ScrollEventAxis, const FloatSize& viewportSize, FloatPoint scrollDestinationOffset, float velocity, std::optional<float> originalPositionForDirectionalSnapping) const;


template <> template <>
FloatScrollSnapOffsetsInfo LayoutScrollSnapOffsetsInfo::convertUnits(float deviceScaleFactor) const;
template <> template <>
WEBCORE_EXPORT std::pair<LayoutUnit, std::optional<unsigned>> LayoutScrollSnapOffsetsInfo::closestSnapOffset(ScrollEventAxis, const LayoutSize& viewportSize, LayoutPoint scrollDestinationOffset, float velocity, std::optional<LayoutUnit> originalPositionForDirectionalSnapping) const;

// Update the snap offsets for this scrollable area, given the RenderBox of the scroll container, the RenderStyle
// which defines the scroll-snap properties, and the viewport rectangle with the origin at the top left of
// the scrolling container's border box.
void updateSnapOffsetsForScrollableArea(ScrollableArea&, const RenderBox& scrollingElementBox, const RenderStyle& scrollingElementStyle, LayoutRect viewportRectInBorderBoxCoordinates, WritingMode, Element*);

template <typename T> WTF::TextStream& operator<<(WTF::TextStream& ts, SnapOffset<T> offset)
{
    ts << offset.offset << " snapTargetID: " <<  offset.snapTargetID << " isFocused: " << offset.isFocused;
    if (offset.stop == ScrollSnapStop::Always)
        ts << " (always)";
    return ts;
}

}; // namespace WebCore
