/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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

#include "RenderStyleConstants.h"
#include <wtf/EnumTraits.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class StyleSelfAlignmentData {
public:
    constexpr StyleSelfAlignmentData() = default;

    // Style data for Self-Aligment and Default-Alignment properties: align-{self, items}, justify-{self, items}.
    // [ <self-position> && <overflow-position>? ] | [ legacy && [ left | right | center ] ]
    constexpr StyleSelfAlignmentData(ItemPosition position, OverflowAlignment overflow = OverflowAlignment::Default, ItemPositionType positionType = ItemPositionType::NonLegacy)
        : m_position(enumToUnderlyingType(position))
        , m_positionType(enumToUnderlyingType(positionType))
        , m_overflow(enumToUnderlyingType(overflow))
    {
    }

    void setPosition(ItemPosition position) { m_position = enumToUnderlyingType(position); }
    void setPositionType(ItemPositionType positionType) { m_positionType = enumToUnderlyingType(positionType); }
    void setOverflow(OverflowAlignment overflow) { m_overflow = enumToUnderlyingType(overflow); }

    ItemPosition position() const { return static_cast<ItemPosition>(m_position); }
    ItemPositionType positionType() const { return static_cast<ItemPositionType>(m_positionType); }
    OverflowAlignment overflow() const { return static_cast<OverflowAlignment>(m_overflow); }

    friend bool operator==(const StyleSelfAlignmentData&, const StyleSelfAlignmentData&) = default;

private:
    uint8_t m_position : 4 { 0 }; // ItemPosition
    uint8_t m_positionType: 1 { 0 }; // ItemPositionType: Whether or not alignment uses the 'legacy' keyword.
    uint8_t m_overflow : 2 { 0 }; // OverflowAlignment
};

WTF::TextStream& operator<<(WTF::TextStream&, const StyleSelfAlignmentData&);

} // namespace WebCore
