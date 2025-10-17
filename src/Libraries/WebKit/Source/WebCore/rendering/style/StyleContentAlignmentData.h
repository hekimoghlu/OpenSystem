/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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
#include "WritingMode.h"

namespace WTF {
class TextStream;
}

namespace WebCore {

class StyleContentAlignmentData {
public:
    constexpr StyleContentAlignmentData() = default;

    // Style data for Content-Distribution properties: align-content, justify-content.
    // <content-distribution> || [ <overflow-position>? && <content-position> ]
    constexpr StyleContentAlignmentData(ContentPosition position, ContentDistribution distribution, OverflowAlignment overflow = OverflowAlignment::Default)
        : m_position(enumToUnderlyingType(position))
        , m_distribution(enumToUnderlyingType(distribution))
        , m_overflow(enumToUnderlyingType(overflow))
    {
    }

    void setPosition(ContentPosition position) { m_position = enumToUnderlyingType(position); }
    void setDistribution(ContentDistribution distribution) { m_distribution = enumToUnderlyingType(distribution); }
    void setOverflow(OverflowAlignment overflow) { m_overflow = enumToUnderlyingType(overflow); }

    ContentPosition position() const { return static_cast<ContentPosition>(m_position); }
    ContentDistribution distribution() const { return static_cast<ContentDistribution>(m_distribution); }
    OverflowAlignment overflow() const { return static_cast<OverflowAlignment>(m_overflow); }
    bool isNormal() const
    {
        return ContentPosition::Normal == static_cast<ContentPosition>(m_position)
        && ContentDistribution::Default == static_cast<ContentDistribution>(m_distribution);
    }
    bool isStartward(std::optional<TextDirection> leftRightAxisDirection = std::nullopt, bool isFlexReverse = false) const;
    bool isEndward(std::optional<TextDirection> leftRightAxisDirection = std::nullopt, bool isFlexReverse = false) const;
    // leftRightAxisDirection is only needed for justify-content (invalid for align-content).
    // Pass std::nullopt if neither the inline axis nor the physical left-right axis matches the justify-content axis (e.g. in flexbox).
    bool isCentered() const;

    friend bool operator==(const StyleContentAlignmentData&, const StyleContentAlignmentData&) = default;

private:
    uint16_t m_position : 4 { 0 }; // ContentPosition
    uint16_t m_distribution : 3 { 0 }; // ContentDistribution
    uint16_t m_overflow : 2 { 0 }; // OverflowAlignment
};

WTF::TextStream& operator<<(WTF::TextStream&, const StyleContentAlignmentData&);

} // namespace WebCore
