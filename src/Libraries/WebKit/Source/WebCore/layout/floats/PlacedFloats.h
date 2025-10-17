/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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

#include "LayoutBoxGeometry.h"
#include "LayoutElementBox.h"
#include "LayoutShape.h"
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace Layout {

class Box;
class BoxGeometry;
class Rect;

// PlacedFloats are relative to the BFC's logical top/left.
// When floats added by nested IFCs with mismatching inline direcctions (e.g. where BFC is RTL but IFC is RTL)
// they get converted as if they had the same inline direction as BFC. What it simply means that
// PlacedFloats::Item::isStartPositioned is always relative to BFC, regardless of what it is relative to in its IFC.
class PlacedFloats {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PlacedFloats);
public:
    PlacedFloats(const ElementBox& blockFormattingContextRoot);

    const ElementBox& blockFormattingContextRoot() const { return m_blockFormattingContextRoot; }

    class Item {
    public:
        // FIXME: This c'tor is only used by the render tree integation codepath.
        enum class Position { Start, End };
        Item(Position, const BoxGeometry& absoluteBoxGeometry, LayoutPoint localTopLeft, const LayoutShape*);
        Item(const Box&, Position, const BoxGeometry& absoluteBoxGeometry, LayoutPoint localTopLeft, std::optional<size_t> line);

        ~Item();

        bool isStartPositioned() const { return m_position == Position::Start; }
        bool isEndPositioned() const { return m_position == Position::End; }
        bool isInFormattingContextOf(const ElementBox& formattingContextRoot) const;

        BoxGeometry boxGeometry() const;

        Rect absoluteRectWithMargin() const { return BoxGeometry::marginBoxRect(m_absoluteBoxGeometry); }
        Rect absoluteBorderBoxRect() const { return BoxGeometry::borderBoxRect(m_absoluteBoxGeometry); }
        BoxGeometry::HorizontalEdges inlineAxisMargin() const { return m_absoluteBoxGeometry.horizontalMargin(); }
        PositionInContextRoot absoluteBottom() const { return { absoluteRectWithMargin().bottom() }; }

        const LayoutShape* shape() const { return m_shape.get(); }
        std::optional<size_t> placedByLine() const { return m_placedByLine; }

        const Box* layoutBox() const { return m_layoutBox.get(); }

    private:
        CheckedPtr<const Box> m_layoutBox;
        Position m_position;
        BoxGeometry m_absoluteBoxGeometry;
        LayoutPoint m_localTopLeft;
        RefPtr<const LayoutShape> m_shape;
        std::optional<size_t> m_placedByLine;
    };
    using List = Vector<Item>;
    const List& list() const { return m_list; }
    const Item* last() const { return list().isEmpty() ? nullptr : &m_list.last(); }

    void append(Item);
    bool remove(const Box&);
    void clear();

    bool isEmpty() const { return list().isEmpty(); }
    bool hasStartPositioned() const;
    bool hasEndPositioned() const;

    std::optional<LayoutUnit> highestPositionOnBlockAxis() const;
    std::optional<LayoutUnit> lowestPositionOnBlockAxis(Clear = Clear::Both) const;

    void shrinkToFit();

private:
    CheckedRef<const ElementBox> m_blockFormattingContextRoot;
    List m_list;
    enum class PositionType {
        Start = 1 << 0,
        End  = 1 << 1
    };
    OptionSet<PositionType> m_positionTypes;
    WritingMode m_writingMode;
};

inline bool PlacedFloats::remove(const Box& floatBox)
{
    return m_list.removeFirstMatching([&floatBox](auto& placedFloatItem) {
        return placedFloatItem.layoutBox() == &floatBox;
    });
}

inline bool PlacedFloats::hasStartPositioned() const
{
    return m_positionTypes.contains(PositionType::Start);
}

inline bool PlacedFloats::hasEndPositioned() const
{
    return m_positionTypes.contains(PositionType::End);
}

}
}
