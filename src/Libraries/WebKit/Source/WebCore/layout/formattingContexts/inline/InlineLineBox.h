/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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

#include "InlineLevelBox.h"
#include "InlineLine.h"
#include "InlineRect.h"
#include "LayoutElementBox.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {
namespace Layout {

class BoxGeometry;
class InlineFormattingContext;
class LineBoxBuilder;
class LineBoxVerticalAligner;
class RubyFormattingContext;

//   ____________________________________________________________ Line Box
// |                                    --------------------
// |                                   |                    |
// | ----------------------------------|--------------------|---------- Root Inline Box
// ||   _____    ___      ___          |                    |
// ||  |        /   \    /   \         |  Inline Level Box  |
// ||  |_____  |     |  |     |        |                    |    ascent
// ||  |       |     |  |     |        |                    |
// ||__|________\___/____\___/_________|____________________|_______ alignment_baseline
// ||
// ||                                                      descent
// ||_______________________________________________________________
// |________________________________________________________________
// The resulting rectangular area that contains the boxes that form a single line of inline-level content is called a line box.
// https://www.w3.org/TR/css-inline-3/#model
class LineBox {
    WTF_MAKE_TZONE_ALLOCATED(LineBox);
public:
    LineBox(const Box& rootLayoutBox, InlineLayoutUnit contentLogicalLeft, InlineLayoutUnit contentLogicalWidth, size_t lineIndex, size_t nonSpanningInlineLevelBoxCount);

    // Note that the line can have many inline boxes and be "empty" the same time e.g. <div><span></span><span></span></div>
    bool hasContent() const { return m_hasContent; }
    bool hasInlineBox() const { return m_boxTypes.contains(InlineLevelBox::Type::InlineBox); }
    bool hasNonInlineBox() const { return m_boxTypes.containsAny({ InlineLevelBox::Type::AtomicInlineBox, InlineLevelBox::Type::LineBreakBox, InlineLevelBox::Type::GenericInlineLevelBox }); }
    bool hasAtomicInlineBox() const { return m_boxTypes.contains(InlineLevelBox::Type::AtomicInlineBox); }

    InlineRect logicalRectForTextRun(const Line::Run&) const;
    InlineRect logicalRectForLineBreakBox(const Box&) const;
    InlineRect logicalRectForRootInlineBox() const { return m_rootInlineBox.logicalRect(); }
    InlineRect logicalBorderBoxForAtomicInlineBox(const Box&, const BoxGeometry&) const;
    InlineRect logicalBorderBoxForInlineBox(const Box&, const BoxGeometry&) const;

    const InlineLevelBox* inlineLevelBoxFor(const Box& layoutBox) const { return const_cast<LineBox&>(*this).inlineLevelBoxFor(layoutBox); }
    const InlineLevelBox& inlineLevelBoxFor(const Line::Run& lineRun) const { return const_cast<LineBox&>(*this).inlineLevelBoxFor(lineRun); }

    const InlineLevelBox& rootInlineBox() const { return m_rootInlineBox; }
    using InlineLevelBoxList = Vector<InlineLevelBox>;
    const InlineLevelBoxList& nonRootInlineLevelBoxes() const { return m_nonRootInlineLevelBoxList; }

    FontBaseline baselineType() const { return m_baselineType; }

    const InlineRect& logicalRect() const { return m_logicalRect; }

    size_t lineIndex() const { return m_lineIndex; }

private:
    friend class LineBoxBuilder;
    friend class LineBoxVerticalAligner;
    friend class RubyFormattingContext;

    void addInlineLevelBox(InlineLevelBox&&);
    InlineLevelBoxList& nonRootInlineLevelBoxes() { return m_nonRootInlineLevelBoxList; }

    InlineLevelBox& rootInlineBox() { return m_rootInlineBox; }

    const InlineLevelBox& parentInlineBox(const InlineLevelBox& inlineLevelBox) const { return const_cast<LineBox&>(*this).parentInlineBox(inlineLevelBox); }
    InlineLevelBox& parentInlineBox(const InlineLevelBox&);

    const InlineLevelBox& parentInlineBox(const Line::Run& lineRun) const { return const_cast<LineBox&>(*this).parentInlineBox(lineRun); }
    InlineLevelBox& parentInlineBox(const Line::Run&);

    InlineLevelBox& inlineLevelBoxFor(const Line::Run&);
    InlineLevelBox* inlineLevelBoxFor(const Box& layoutBox);

    InlineRect logicalRectForInlineLevelBox(const Box& layoutBox) const;

    void setLogicalRect(const InlineRect& logicalRect) { m_logicalRect = logicalRect; }
    void setHasContent(bool hasContent) { m_hasContent = hasContent; }
    void setBaselineType(FontBaseline baselineType) { m_baselineType = baselineType; }

    InlineLayoutUnit inlineLevelBoxAbsoluteTop(const InlineLevelBox&) const;

private:
    size_t m_lineIndex { 0 };
    bool m_hasContent { false };
    InlineRect m_logicalRect;
    OptionSet<InlineLevelBox::Type> m_boxTypes;

    FontBaseline m_baselineType { AlphabeticBaseline };
    InlineLevelBox m_rootInlineBox;
    InlineLevelBoxList m_nonRootInlineLevelBoxList;

    UncheckedKeyHashMap<const Box*, size_t> m_nonRootInlineLevelBoxMap;
};

inline InlineLevelBox* LineBox::inlineLevelBoxFor(const Box& layoutBox)
{
    if (&layoutBox == &m_rootInlineBox.layoutBox())
        return &m_rootInlineBox;
    auto entry = m_nonRootInlineLevelBoxMap.find(&layoutBox);
    if (entry == m_nonRootInlineLevelBoxMap.end())
        return nullptr;
    return &m_nonRootInlineLevelBoxList[entry->value];
}

inline InlineLevelBox& LineBox::parentInlineBox(const InlineLevelBox& inlineLevelBox)
{
    return *inlineLevelBoxFor(inlineLevelBox.layoutBox().parent());
}

inline InlineLevelBox& LineBox::parentInlineBox(const Line::Run& lineRun)
{
    return *inlineLevelBoxFor(lineRun.layoutBox().parent());
}

inline InlineLevelBox& LineBox::inlineLevelBoxFor(const Line::Run& lineRun)
{
    return *inlineLevelBoxFor(lineRun.layoutBox());
}

}
}

