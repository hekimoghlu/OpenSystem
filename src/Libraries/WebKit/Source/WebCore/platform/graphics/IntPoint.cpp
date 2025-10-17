/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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
#include "IntPoint.h"

#include "FloatPoint.h"
#include "IntRect.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

IntPoint::IntPoint(const FloatPoint& p)
    : m_x(clampToInteger(p.x()))
    , m_y(clampToInteger(p.y()))
{
}

IntPoint IntPoint::constrainedBetween(const IntPoint& min, const IntPoint& max) const
{
    return {
        std::max(min.x(), std::min(max.x(), m_x)),
        std::max(min.y(), std::min(max.y(), m_y))
    };
}

IntPoint IntPoint::constrainedWithin(const IntRect& rect) const
{
    return constrainedBetween(rect.minXMinYCorner(), rect.maxXMaxYCorner());
}

TextStream& operator<<(TextStream& ts, const IntPoint& p)
{
    return ts << "(" << p.x() << "," << p.y() << ")";
}

}
