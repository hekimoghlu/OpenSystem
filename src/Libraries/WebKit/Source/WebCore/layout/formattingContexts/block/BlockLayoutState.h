/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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

#include "Font.h"
#include "PlacedFloats.h"
#include "StyleTextEdge.h"
#include <algorithm>

namespace WebCore {
namespace Layout {

class BlockFormattingContext;

// This class holds block level information shared across child inline formatting contexts.
class BlockLayoutState {
public:
    struct LineClamp {
        size_t maximumLines { 0 };
        bool shouldDiscardOverflow { false };
        bool isLegacy { true };
    };
    enum class TextBoxTrimSide : uint8_t {
        Start = 1 << 0,
        End   = 1 << 1
    };
    using TextBoxTrim = OptionSet<TextBoxTrimSide>;

    struct LineGrid {
        LayoutSize layoutOffset;
        LayoutSize gridOffset;
        InlineLayoutUnit columnWidth;
        LayoutUnit rowHeight;
        LayoutUnit topRowOffset;
        Ref<const Font> primaryFont;
        std::optional<LayoutSize> paginationOrigin;
        LayoutUnit pageLogicalTop;
    };

    BlockLayoutState(PlacedFloats&, std::optional<LineClamp> = { }, TextBoxTrim = { }, TextEdge = { }, std::optional<LayoutUnit> intrusiveInitialLetterLogicalBottom = { }, std::optional<LineGrid> lineGrid = { });

    PlacedFloats& placedFloats() { return m_placedFloats; }
    const PlacedFloats& placedFloats() const { return m_placedFloats; }

    std::optional<LineClamp> lineClamp() const { return m_lineClamp; }
    TextBoxTrim textBoxTrim() const { return m_textBoxTrim; }
    TextEdge textBoxEdge() const { return m_textBoxEdge; }

    std::optional<LayoutUnit> intrusiveInitialLetterLogicalBottom() const { return m_intrusiveInitialLetterLogicalBottom; }
    const std::optional<LineGrid>& lineGrid() const { return m_lineGrid; }

private:
    PlacedFloats& m_placedFloats;
    std::optional<LineClamp> m_lineClamp;
    TextBoxTrim m_textBoxTrim;
    TextEdge m_textBoxEdge;
    std::optional<LayoutUnit> m_intrusiveInitialLetterLogicalBottom;
    std::optional<LineGrid> m_lineGrid;
};

inline BlockLayoutState::BlockLayoutState(PlacedFloats& placedFloats, std::optional<LineClamp> lineClamp, TextBoxTrim textBoxTrim, TextEdge textBoxEdge, std::optional<LayoutUnit> intrusiveInitialLetterLogicalBottom, std::optional<LineGrid> lineGrid)
    : m_placedFloats(placedFloats)
    , m_lineClamp(lineClamp)
    , m_textBoxTrim(textBoxTrim)
    , m_textBoxEdge(textBoxEdge)
    , m_intrusiveInitialLetterLogicalBottom(intrusiveInitialLetterLogicalBottom)
    , m_lineGrid(lineGrid)
{
}

}
}
