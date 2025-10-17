/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#include "FloatLine.h"

namespace WebCore {

const FloatPoint FloatLine::pointAtAbsoluteDistance(float absoluteDistance) const
{
    if (!length())
        return m_start;
    auto relativeDistance = absoluteDistance / length();
    return pointAtRelativeDistance(relativeDistance);
}

const FloatPoint FloatLine::pointAtRelativeDistance(float relativeDistance) const
{
    return {
        m_start.x() - (relativeDistance * (m_start.x() - m_end.x())),
        m_start.y() - (relativeDistance * (m_start.y() - m_end.y())),
    };
}

const FloatLine FloatLine::extendedToBounds(const FloatRect& bounds) const
{
    if (std::abs(m_start.x() - m_end.x()) <= std::abs(m_start.y() - m_end.y())) {
        // The line is roughly vertical, so construct points at the top and bottom of the bounds.
        FloatPoint top = { (((bounds.y() - m_start.y()) * (m_end.x() - m_start.x())) / (m_end.y() - m_start.y())) + m_start.x(), bounds.y() };
        FloatPoint bottom = { (((bounds.y() + bounds.height() - m_start.y()) * (m_end.x() - m_start.x())) / (m_end.y() - m_start.y())) + m_start.x(), bounds.y() + bounds.height() };
        return { top, bottom };
    }
    
    // The line is roughly horizontal, so construct points at the left and right of the bounds.
    FloatPoint left = { bounds.x(), (((bounds.x() - m_start.x()) * (m_end.y() - m_start.y())) / (m_end.x() - m_start.x())) + m_start.y() };
    FloatPoint right = { bounds.x() + bounds.width(), (((bounds.x() + bounds.width() - m_start.x()) * (m_end.y() - m_start.y())) / (m_end.x() - m_start.x())) + m_start.y() };
    return { left, right };
}

const std::optional<FloatPoint> FloatLine::intersectionWith(const FloatLine& otherLine) const
{
    float denominator = ((m_start.x() - m_end.x()) * (otherLine.start().y() - otherLine.end().y())) - ((m_start.y() - m_end.y()) * (otherLine.start().x() - otherLine.end().x()));
    
    // A denominator of zero indicates the lines are parallel or coincident, which means there is no true intersection.
    if (!denominator)
        return std::nullopt;
    
    float thisLineCommonNumeratorFactor = (m_start.x() * m_end.y()) - (m_start.y() * m_end.x());
    float otherLineCommonNumeratorFactor = (otherLine.start().x() * otherLine.end().y()) - (otherLine.start().y() * otherLine.end().x());
    
    return {{
        ((thisLineCommonNumeratorFactor * (otherLine.start().x() - otherLine.end().x())) - (otherLineCommonNumeratorFactor * (m_start.x() - m_end.x()))) / denominator,
        ((thisLineCommonNumeratorFactor * (otherLine.start().y() - otherLine.end().y())) - (otherLineCommonNumeratorFactor * (m_start.y() - m_end.y()))) / denominator,
    }};
}

}
