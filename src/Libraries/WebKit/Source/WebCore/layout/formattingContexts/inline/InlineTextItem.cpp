/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
#include "config.h"
#include "InlineTextItem.h"

#include "FontCascade.h"
#include "InlineSoftLineBreakItem.h"
#include "RenderStyleInlines.h"
#include "TextUtil.h"
#include <wtf/unicode/CharacterNames.h>

namespace WebCore {
namespace Layout {

static_assert(sizeof(InlineItem) == sizeof(InlineTextItem));

InlineTextItem::InlineTextItem(const InlineTextBox& inlineTextBox, unsigned start, unsigned length, UBiDiLevel bidiLevel, bool hasTrailingSoftHyphen, bool isWordSeparator, std::optional<InlineLayoutUnit> width, TextItemType textItemType)
    : InlineItem(inlineTextBox, Type::Text, bidiLevel)
{
    m_startOrPosition = start;
    m_length = length;
    m_hasWidth = !!width;
    m_hasTrailingSoftHyphen = hasTrailingSoftHyphen;
    m_isWordSeparator = isWordSeparator;
    m_width = width.value_or(0);
    m_textItemType = textItemType;
}

InlineTextItem::InlineTextItem(const InlineTextBox& inlineTextBox)
    : InlineItem(inlineTextBox, Type::Text, UBIDI_DEFAULT_LTR)
{
}

InlineTextItem InlineTextItem::left(unsigned length) const
{
    RELEASE_ASSERT(length <= this->length());
    ASSERT(m_textItemType != TextItemType::Undefined);
    ASSERT(length);
    return { inlineTextBox(), start(), length, bidiLevel(), false, isWordSeparator(), std::nullopt, m_textItemType };
}

InlineTextItem InlineTextItem::right(unsigned length, std::optional<InlineLayoutUnit> width) const
{
    RELEASE_ASSERT(length <= this->length());
    ASSERT(m_textItemType != TextItemType::Undefined);
    ASSERT(length);
    return { inlineTextBox(), end() - length, length, bidiLevel(), hasTrailingSoftHyphen(), isWordSeparator(), width, m_textItemType };
}

InlineTextItem InlineTextItem::split(size_t leftSideLength)
{
    RELEASE_ASSERT(length() > 1);
    RELEASE_ASSERT(leftSideLength && leftSideLength < length());
    auto rightSide = right(length() - leftSideLength, { });
    m_length = length() - rightSide.length();
    m_hasWidth = false;
    m_width = { };
    return rightSide;
}

bool InlineTextItem::isZeroWidthSpaceSeparator() const
{
    // FIXME: We should check for more zero width content and not just U+200B.
    return !m_length || (m_length == 1 && inlineTextBox().content()[start()] == zeroWidthSpace); 
}

bool InlineTextItem::isQuirkNonBreakingSpace() const
{
    if (style().nbspMode() != NBSPMode::Space || style().textWrapMode() == TextWrapMode::NoWrap || style().whiteSpaceCollapse() == WhiteSpaceCollapse::BreakSpaces)
        return false;
    return m_length && inlineTextBox().content()[start()] == noBreakSpace;
}

bool InlineTextItem::isFullyTrimmable() const
{
    return isWhitespace() && !TextUtil::shouldPreserveSpacesAndTabs(layoutBox());
}

bool InlineTextItem::shouldPreserveSpacesAndTabs(const InlineTextItem& inlineTextItem)
{
    ASSERT(inlineTextItem.isWhitespace());
    return TextUtil::shouldPreserveSpacesAndTabs(inlineTextItem.layoutBox());
}

}
}
