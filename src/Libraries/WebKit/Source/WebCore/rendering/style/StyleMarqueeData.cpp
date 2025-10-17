/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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
#include "StyleMarqueeData.h"

#include "RenderStyleConstants.h"
#include "RenderStyleDifference.h"
#include "RenderStyleInlines.h"

namespace WebCore {

StyleMarqueeData::StyleMarqueeData()
    : increment(RenderStyle::initialMarqueeIncrement())
    , speed(RenderStyle::initialMarqueeSpeed())
    , loops(RenderStyle::initialMarqueeLoopCount())
    , behavior(static_cast<unsigned>(RenderStyle::initialMarqueeBehavior()))
    , direction(static_cast<unsigned>(RenderStyle::initialMarqueeDirection()))
{
}

inline StyleMarqueeData::StyleMarqueeData(const StyleMarqueeData& o)
    : RefCounted<StyleMarqueeData>()
    , increment(o.increment)
    , speed(o.speed)
    , loops(o.loops)
    , behavior(o.behavior)
    , direction(o.direction) 
{
}

Ref<StyleMarqueeData> StyleMarqueeData::create()
{
    return adoptRef(*new StyleMarqueeData);
}

Ref<StyleMarqueeData> StyleMarqueeData::copy() const
{
    return adoptRef(*new StyleMarqueeData(*this));
}

bool StyleMarqueeData::operator==(const StyleMarqueeData& o) const
{
    return increment == o.increment && speed == o.speed && direction == o.direction &&
           behavior == o.behavior && loops == o.loops;
}

#if !LOG_DISABLED
void StyleMarqueeData::dumpDifferences(TextStream& ts, const StyleMarqueeData& other) const
{
    LOG_IF_DIFFERENT(increment);
    LOG_IF_DIFFERENT(speed);
    LOG_IF_DIFFERENT_WITH_CAST(MarqueeBehavior, behavior);
    LOG_IF_DIFFERENT_WITH_CAST(MarqueeDirection, direction);
}
#endif // !LOG_DISABLED

} // namespace WebCore
