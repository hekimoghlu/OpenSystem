/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
#include "KeyboardScroll.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

FloatSize unitVectorForScrollDirection(ScrollDirection direction)
{
    switch (direction) {
    case ScrollDirection::ScrollUp:
        return { 0, -1 };
    case ScrollDirection::ScrollDown:
        return { 0, 1 };
    case ScrollDirection::ScrollLeft:
        return { -1, 0 };
    case ScrollDirection::ScrollRight:
        return { 1, 0 };
    }

    RELEASE_ASSERT_NOT_REACHED();
}

TextStream& operator<<(TextStream& ts, const KeyboardScroll& scrollData)
{
    return ts << "offset=" << scrollData.offset << " maximumVelocity=" << scrollData.maximumVelocity << " force=" << scrollData.force << " granularity=" << scrollData.granularity << " direction=" << scrollData.direction;
}

TextStream& operator<<(TextStream& ts, const KeyboardScrollParameters& parameters)
{
    return ts << "springMass=" << parameters.springMass << " springStiffness=" << parameters.springStiffness
    << " springDamping=" << parameters.springDamping << " maximumVelocityMultiplier=" << parameters.maximumVelocityMultiplier
    << " timeToMaximumVelocity=" << parameters.timeToMaximumVelocity << " rubberBandForce=" << parameters.rubberBandForce;
}

}
