/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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

#include "LayoutUnit.h"
#include "LayoutPoint.h"
#include "LayoutRect.h"
#include "MarginTypes.h"
#include <wtf/HashFunctions.h>
#include <wtf/HashTraits.h>

namespace WebCore {

namespace Layout {

#define USE_FLOAT_AS_INLINE_LAYOUT_UNIT 1

#if USE_FLOAT_AS_INLINE_LAYOUT_UNIT
using InlineLayoutUnit = float;
using InlineLayoutPoint = FloatPoint;
using InlineLayoutSize = FloatSize;
using InlineLayoutRect = FloatRect;
#else
using InlineLayoutUnit = LayoutUnit;
using InlineLayoutPoint = LayoutPoint;
using InlineLayoutSize = LayoutSize;
using InlineLayoutRect = LayoutRect;
#endif

struct Position {
    operator LayoutUnit() const { return value; }
    friend bool operator==(Position, Position) = default;

    LayoutUnit value;
};

inline bool operator<(const Position& a, const Position& b)
{
    return a.value < b.value;
}

struct Point {
    // FIXME: Use Position<Horizontal>, Position<Vertical> to avoid top/left vs. x/y confusion.
    LayoutUnit x; // left
    LayoutUnit y; // top

    Point() = default;
    Point(LayoutUnit, LayoutUnit);
    Point(LayoutPoint);
    static Point max() { return { LayoutUnit::max(), LayoutUnit::max() }; }

    void move(LayoutSize);
    void moveBy(LayoutPoint);
    operator LayoutPoint() const { return { x, y }; }
};

// FIXME: Wrap these into structs.
using PointInContextRoot = Point;
using PositionInContextRoot = Position;

inline Point::Point(LayoutPoint point)
    : x(point.x())
    , y(point.y())
{
}

inline Point::Point(LayoutUnit x, LayoutUnit y)
    : x(x)
    , y(y)
{
}

inline void Point::move(LayoutSize offset)
{
    x += offset.width();
    y += offset.height();
}

inline void Point::moveBy(LayoutPoint offset)
{
    x += offset.x();
    y += offset.y();
}

struct ContentWidthAndMargin {
    LayoutUnit contentWidth;
    UsedHorizontalMargin usedMargin;
};

struct ContentHeightAndMargin {
    LayoutUnit contentHeight;
    UsedVerticalMargin::NonCollapsedValues nonCollapsedMargin;
};

struct HorizontalGeometry {
    LayoutUnit left;
    LayoutUnit right;
    ContentWidthAndMargin contentWidthAndMargin;
};

struct VerticalGeometry {
    LayoutUnit top;
    LayoutUnit bottom;
    ContentHeightAndMargin contentHeightAndMargin;
};

struct OverriddenHorizontalValues {
    std::optional<LayoutUnit> width;
    std::optional<UsedHorizontalMargin> margin;
};

struct OverriddenVerticalValues {
    // Consider collapsing it.
    std::optional<LayoutUnit> height;
};

inline LayoutUnit toLayoutUnit(InlineLayoutUnit value)
{
    return LayoutUnit { value };
}

inline LayoutUnit ceiledLayoutUnit(InlineLayoutUnit value)
{
    return LayoutUnit::fromFloatCeil(value);
}

inline LayoutPoint toLayoutPoint(const InlineLayoutPoint& point)
{
    return LayoutPoint { point };
}

inline LayoutSize toLayoutSize(const InlineLayoutSize& size)
{
    return LayoutSize { size };
}

inline LayoutRect toLayoutRect(const InlineLayoutRect& rect)
{
    return LayoutRect { rect };
}

inline InlineLayoutUnit maxInlineLayoutUnit()
{
#if USE_FLOAT_AS_INLINE_LAYOUT_UNIT
    return std::numeric_limits<float>::max();
#else
    return LayoutUnit::max();
#endif
}

struct SlotPosition {
    SlotPosition() = default;
    SlotPosition(size_t column, size_t row);

    friend bool operator==(const SlotPosition&, const SlotPosition&) = default;

    size_t column { 0 };
    size_t row { 0 };
};

inline SlotPosition::SlotPosition(size_t column, size_t row)
    : column(column)
    , row(row)
{
}

struct CellSpan {
    size_t column { 1 };
    size_t row { 1 };
};

}
}

namespace WTF {
struct SlotPositionHash {
    static unsigned hash(const WebCore::Layout::SlotPosition& slotPosition) { return pairIntHash(slotPosition.column, slotPosition.row); }
    static bool equal(const WebCore::Layout::SlotPosition& a, const WebCore::Layout::SlotPosition& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};
template<> struct HashTraits<WebCore::Layout::SlotPosition> : GenericHashTraits<WebCore::Layout::SlotPosition> {
    static WebCore::Layout::SlotPosition emptyValue() { return WebCore::Layout::SlotPosition(std::numeric_limits<size_t>::max() - 1, std::numeric_limits<size_t>::max() - 1); }
    static bool isEmptyValue(const WebCore::Layout::SlotPosition& value) { return value.column == (std::numeric_limits<size_t>::max() - 1); }

    static void constructDeletedValue(WebCore::Layout::SlotPosition& slot) { slot.column = std::numeric_limits<size_t>::max(); }
    static bool isDeletedValue(const WebCore::Layout::SlotPosition& slot) { return slot.column == std::numeric_limits<size_t>::max(); }
};
template<> struct DefaultHash<WebCore::Layout::SlotPosition> : SlotPositionHash { };
}

