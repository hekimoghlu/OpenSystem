/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#include "ScrollAlignment.h"

#include "Logging.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

const ScrollAlignment ScrollAlignment::alignCenterIfNotVisible = { Behavior::NoScroll, Behavior::AlignCenter, Behavior::NoScroll };
const ScrollAlignment ScrollAlignment::alignToEdgeIfNotVisible = { Behavior::NoScroll, Behavior::AlignToClosestEdge, Behavior::NoScroll };
const ScrollAlignment ScrollAlignment::alignCenterIfNeeded = { Behavior::NoScroll, Behavior::AlignCenter, Behavior::AlignToClosestEdge };
WEBCORE_EXPORT const ScrollAlignment ScrollAlignment::alignToEdgeIfNeeded = { Behavior::NoScroll, Behavior::AlignToClosestEdge, Behavior::AlignToClosestEdge };
WEBCORE_EXPORT const ScrollAlignment ScrollAlignment::alignCenterAlways = { Behavior::AlignCenter, Behavior::AlignCenter, Behavior::AlignCenter };
const ScrollAlignment ScrollAlignment::alignTopAlways = { Behavior::AlignTop, Behavior::AlignTop, Behavior::AlignTop };
const ScrollAlignment ScrollAlignment::alignRightAlways = { Behavior::AlignRight, Behavior::AlignRight, Behavior::AlignRight };
const ScrollAlignment ScrollAlignment::alignLeftAlways = { Behavior::AlignLeft, Behavior::AlignLeft, Behavior::AlignLeft };
const ScrollAlignment ScrollAlignment::alignBottomAlways = { Behavior::AlignBottom, Behavior::AlignBottom, Behavior::AlignBottom };
    
TextStream& operator<<(TextStream& ts, ScrollAlignment::Behavior b)
{
    switch (b) {
    case ScrollAlignment::Behavior::NoScroll:
        return ts << "NoScroll";
    case ScrollAlignment::Behavior::AlignCenter:
        return ts << "AlignCenter";
    case ScrollAlignment::Behavior::AlignTop:
        return ts << "AlignTop";
    case ScrollAlignment::Behavior::AlignBottom:
        return ts << "AlignBottom";
    case ScrollAlignment::Behavior::AlignLeft:
        return ts << "AlignLeft";
    case ScrollAlignment::Behavior::AlignRight:
        return ts << "AlignRight";
    case ScrollAlignment::Behavior::AlignToClosestEdge:
        return ts << "AlignToClosestEdge";
    }
    return ts;
}
    
TextStream& operator<<(TextStream& ts, const ScrollAlignment& s)
{
    return ts << "ScrollAlignment: visible: " << s.m_rectVisible << " hidden: " << s.m_rectHidden << " partial: " << s.m_rectPartial;
}

}; // namespace WebCore
