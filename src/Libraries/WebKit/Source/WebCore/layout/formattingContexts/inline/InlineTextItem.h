/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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

#include "InlineItem.h"
#include "LayoutInlineTextBox.h"

namespace WebCore {
namespace Layout {

class InlineItemsBuilder;

class InlineTextItem : public InlineItem {
public:
    static InlineTextItem createWhitespaceItem(const InlineTextBox&, unsigned start, unsigned length, UBiDiLevel, bool isWordSeparator, std::optional<InlineLayoutUnit> width);
    static InlineTextItem createNonWhitespaceItem(const InlineTextBox&, unsigned start, unsigned length, UBiDiLevel, bool hasTrailingSoftHyphen, std::optional<InlineLayoutUnit> width);
    static InlineTextItem createEmptyItem(const InlineTextBox&);

    unsigned start() const { return m_startOrPosition; }
    unsigned end() const { return start() + length(); }
    unsigned length() const { return m_length; }

    bool isEmpty() const { return !length() && m_textItemType == TextItemType::Undefined; }
    bool isWhitespace() const { return m_textItemType == TextItemType::Whitespace; }
    bool isWordSeparator() const { return m_isWordSeparator; }
    bool isZeroWidthSpaceSeparator() const;
    bool isQuirkNonBreakingSpace() const;
    bool isFullyTrimmable() const;
    bool hasTrailingSoftHyphen() const { return m_hasTrailingSoftHyphen; }
    std::optional<InlineLayoutUnit> width() const { return m_hasWidth ? std::make_optional(m_width) : std::optional<InlineLayoutUnit> { }; }

    const InlineTextBox& inlineTextBox() const { return downcast<InlineTextBox>(layoutBox()); }

    InlineTextItem left(unsigned length) const;
    InlineTextItem right(unsigned length, std::optional<InlineLayoutUnit> width) const;

    static bool shouldPreserveSpacesAndTabs(const InlineTextItem&);

private:
    friend class InlineItemsBuilder;
    using InlineItem::TextItemType;

    InlineTextItem split(size_t leftSideLength);

    InlineTextItem(const InlineTextBox&, unsigned start, unsigned length, UBiDiLevel, bool hasTrailingSoftHyphen, bool isWordSeparator, std::optional<InlineLayoutUnit> width, TextItemType);
    explicit InlineTextItem(const InlineTextBox&);
};

inline InlineTextItem InlineTextItem::createWhitespaceItem(const InlineTextBox& inlineTextBox, unsigned start, unsigned length, UBiDiLevel bidiLevel, bool isWordSeparator, std::optional<InlineLayoutUnit> width)
{
    return { inlineTextBox, start, length, bidiLevel, false, isWordSeparator, width, TextItemType::Whitespace };
}

inline InlineTextItem InlineTextItem::createNonWhitespaceItem(const InlineTextBox& inlineTextBox, unsigned start, unsigned length, UBiDiLevel bidiLevel, bool hasTrailingSoftHyphen, std::optional<InlineLayoutUnit> width)
{
    // FIXME: Use the following list of non-whitespace characters to set the "isWordSeparator" bit: noBreakSpace, ethiopicWordspace, aegeanWordSeparatorLine aegeanWordSeparatorDot ugariticWordDivider.
    return { inlineTextBox, start, length, bidiLevel, hasTrailingSoftHyphen, false, width, TextItemType::NonWhitespace };
}

inline InlineTextItem InlineTextItem::createEmptyItem(const InlineTextBox& inlineTextBox)
{
    ASSERT(!inlineTextBox.content().length());
    return InlineTextItem { inlineTextBox };
}

}
}

SPECIALIZE_TYPE_TRAITS_INLINE_ITEM(InlineTextItem, isText())

