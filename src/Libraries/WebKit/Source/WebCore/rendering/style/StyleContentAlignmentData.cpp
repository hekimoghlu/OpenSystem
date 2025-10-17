/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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
#include "StyleContentAlignmentData.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

TextStream& operator<<(TextStream& ts, const StyleContentAlignmentData& o)
{
    return ts << o.position() << " " << o.distribution() << " " << o.overflow();
}

bool StyleContentAlignmentData::isStartward(std::optional<TextDirection> leftRightAxisDirection, bool isFlexReverse) const
{
    switch (static_cast<ContentPosition>(m_position)) {
    case ContentPosition::Normal:
        switch (static_cast<ContentDistribution>(m_distribution)) {
        case ContentDistribution::Default:
        case ContentDistribution::Stretch:
        case ContentDistribution::SpaceBetween:
            return !isFlexReverse;
        default:
            return false;
        }
    case ContentPosition::Start:
    case ContentPosition::Baseline:
        return true;
    case ContentPosition::End:
    case ContentPosition::LastBaseline:
    case ContentPosition::Center:
        return false;
    case ContentPosition::FlexStart:
        return !isFlexReverse;
    case ContentPosition::FlexEnd:
        return isFlexReverse;
    case ContentPosition::Left:
        if (leftRightAxisDirection)
            return leftRightAxisDirection == TextDirection::LTR;
        return true;
    case ContentPosition::Right:
        if (leftRightAxisDirection)
            return leftRightAxisDirection == TextDirection::RTL;
        return true;
    default:
        ASSERT("Invalid ContentPosition");
        return true;
    }
}
bool StyleContentAlignmentData::isEndward(std::optional<TextDirection> leftRightAxisDirection, bool isFlexReverse) const
{
    switch (static_cast<ContentPosition>(m_position)) {
    case ContentPosition::Normal:
        switch (static_cast<ContentDistribution>(m_distribution)) {
        case ContentDistribution::Default:
        case ContentDistribution::Stretch:
        case ContentDistribution::SpaceBetween:
            return isFlexReverse;
        default:
            return false;
        }
    case ContentPosition::Start:
    case ContentPosition::Baseline:
    case ContentPosition::Center:
        return false;
    case ContentPosition::End:
    case ContentPosition::LastBaseline:
        return true;
    case ContentPosition::FlexStart:
        return isFlexReverse;
    case ContentPosition::FlexEnd:
        return !isFlexReverse;
    case ContentPosition::Left:
        if (leftRightAxisDirection)
            return leftRightAxisDirection == TextDirection::RTL;
        return false;
    case ContentPosition::Right:
        if (leftRightAxisDirection)
            return leftRightAxisDirection == TextDirection::LTR;
        return false;
    default:
        ASSERT("Invalid ContentPosition");
        return false;
    }
}

bool StyleContentAlignmentData::isCentered() const
{
    return static_cast<ContentPosition>(m_position) == ContentPosition::Center
        || static_cast<ContentDistribution>(m_distribution) == ContentDistribution::SpaceAround
        || static_cast<ContentDistribution>(m_distribution) == ContentDistribution::SpaceEvenly;
}

}
