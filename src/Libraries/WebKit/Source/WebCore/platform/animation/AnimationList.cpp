/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#include "AnimationList.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

#define FILL_UNSET_PROPERTY(test, propGet, propSet) \
for (i = 0; i < size() && animation(i).test(); ++i) { } \
if (i) { \
    for (size_t j = 0; i < size(); ++i, ++j) \
        animation(i).propSet(animation(j).propGet()); \
}

AnimationList::AnimationList() = default;

AnimationList::AnimationList(const AnimationList& other, CopyBehavior copyBehavior)
{
    if (copyBehavior == CopyBehavior::Reference)
        m_animations = other.m_animations;
    else {
        m_animations = other.m_animations.map([](auto& animation) {
            return Animation::create(animation.get());
        });
    }
}

void AnimationList::fillUnsetProperties()
{
    size_t i;
    FILL_UNSET_PROPERTY(isDelaySet, delay, fillDelay);
    FILL_UNSET_PROPERTY(isDirectionSet, direction, fillDirection);
    FILL_UNSET_PROPERTY(isDurationSet, duration, fillDuration);
    FILL_UNSET_PROPERTY(isFillModeSet, fillMode, fillFillMode);
    FILL_UNSET_PROPERTY(isIterationCountSet, iterationCount, fillIterationCount);
    FILL_UNSET_PROPERTY(isPlayStateSet, playState, fillPlayState);
    FILL_UNSET_PROPERTY(isTimelineSet, timeline, fillTimeline);
    FILL_UNSET_PROPERTY(isTimingFunctionSet, timingFunction, fillTimingFunction);
    FILL_UNSET_PROPERTY(isPropertySet, property, fillProperty);
    FILL_UNSET_PROPERTY(isCompositeOperationSet, compositeOperation, fillCompositeOperation);
    FILL_UNSET_PROPERTY(isAllowsDiscreteTransitionsSet, allowsDiscreteTransitions, fillAllowsDiscreteTransitions);
    FILL_UNSET_PROPERTY(isRangeStartSet, rangeStart, fillRangeStart);
    FILL_UNSET_PROPERTY(isRangeEndSet, rangeEnd, fillRangeEnd);
}

bool AnimationList::operator==(const AnimationList& other) const
{
    if (size() != other.size())
        return false;
    for (size_t i = 0; i < size(); ++i) {
        if (animation(i) != other.animation(i))
            return false;
    }
    return true;
}

TextStream& operator<<(TextStream& ts, const AnimationList& animationList)
{
    ts << "[";
    for (size_t i = 0; i < animationList.size(); ++i) {
        if (i > 0)
            ts << ", ";
        ts << animationList.animation(i);
    }
    ts << "]";
    return ts;
}

} // namespace WebCore
