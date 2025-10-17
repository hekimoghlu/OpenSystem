/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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

#include "LayoutBox.h"
#include "LayoutUnits.h"
#include <unicode/ubidi.h>

namespace WebCore {
namespace Layout {

class InlineItemsBuilder;

class InlineItem {
public:
    enum class Type : uint8_t {
        Text,
        HardLineBreak,
        SoftLineBreak,
        WordBreakOpportunity,
        AtomicInlineBox,
        InlineBoxStart,
        InlineBoxEnd,
        Float,
        Opaque
    };
    InlineItem(const Box& layoutBox, Type, UBiDiLevel = UBIDI_DEFAULT_LTR);

    Type type() const { return m_type; }
    static constexpr UBiDiLevel opaqueBidiLevel = 0xff;
    UBiDiLevel bidiLevel() const { return m_bidiLevel; }
    const Box& layoutBox() const { return *m_layoutBox; }
    const RenderStyle& style() const { return layoutBox().style(); }
    const RenderStyle& firstLineStyle() const { return layoutBox().firstLineStyle(); }

    bool isText() const { return type() == Type::Text; }
    bool isAtomicInlineBox() const { return type() == Type::AtomicInlineBox; }
    bool isFloat() const { return type() == Type::Float; }
    bool isLineBreak() const { return isSoftLineBreak() || isHardLineBreak(); }
    bool isWordBreakOpportunity() const { return type() == Type::WordBreakOpportunity; }
    bool isSoftLineBreak() const { return type() == Type::SoftLineBreak; }
    bool isHardLineBreak() const { return type() == Type::HardLineBreak; }
    bool isInlineBoxStart() const { return type() == Type::InlineBoxStart; }
    bool isInlineBoxEnd() const { return type() == Type::InlineBoxEnd; }
    bool isInlineBoxStartOrEnd() const { return isInlineBoxStart() || isInlineBoxEnd(); }
    bool isOpaque() const { return type() == Type::Opaque; }

private:
    friend class InlineItemsBuilder;

    void setBidiLevel(UBiDiLevel bidiLevel) { m_bidiLevel = bidiLevel; }
    void setWidth(InlineLayoutUnit);

    const Box* m_layoutBox { nullptr };

protected:
    InlineLayoutUnit m_width { };
    unsigned m_length { 0 };

    // For InlineTextItem and InlineSoftLineBreakItem
    unsigned m_startOrPosition { 0 };
private:
    UBiDiLevel m_bidiLevel { UBIDI_DEFAULT_LTR };

    Type m_type : 4 { };

protected:
    // For InlineTextItem
    enum class TextItemType  : uint8_t { Undefined, Whitespace, NonWhitespace };

    TextItemType m_textItemType : 2 { TextItemType::Undefined };
    bool m_hasWidth : 1 { false };
    bool m_hasTrailingSoftHyphen : 1 { false };
    bool m_isWordSeparator : 1 { false };
};

inline InlineItem::InlineItem(const Box& layoutBox, Type type, UBiDiLevel bidiLevel)
    : m_layoutBox(&layoutBox)
    , m_bidiLevel(bidiLevel)
    , m_type(type)
{
}

inline void InlineItem::setWidth(InlineLayoutUnit width)
{
    m_width = width;
    m_hasWidth = true;
}

using InlineItemList = Vector<InlineItem>;

#define SPECIALIZE_TYPE_TRAITS_INLINE_ITEM(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::Layout::ToValueTypeName) \
    static bool isType(const WebCore::Layout::InlineItem& inlineItem) { return inlineItem.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()

}
}

namespace WTF {

template<>
struct VectorTraits<WebCore::Layout::InlineItem> : public VectorTraitsBase<false, void> {
    static constexpr bool canCopyWithMemcpy = true;
    static constexpr bool canMoveWithMemcpy = true;
};

}
