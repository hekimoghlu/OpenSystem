/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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

#include "FloatSize.h"
#include "ScrollTypes.h"

namespace WTF {
class TextStream;
}

namespace WebCore {

WEBCORE_EXPORT FloatSize unitVectorForScrollDirection(ScrollDirection);

struct KeyboardScroll {
    FloatSize offset; // Points per increment.
    FloatSize maximumVelocity; // Points per second.
    FloatSize force;

    ScrollGranularity granularity { ScrollGranularity::Line };
    ScrollDirection direction { ScrollDirection::ScrollUp };

    friend bool operator==(const KeyboardScroll&, const KeyboardScroll&) = default;
};

struct KeyboardScrollParameters {
    const float springMass;
    const float springStiffness;
    const float springDamping;

    const float maximumVelocityMultiplier;
    const float timeToMaximumVelocity;

    const float rubberBandForce;

    static constexpr KeyboardScrollParameters parameters()
    {
#if PLATFORM(MAC) || PLATFORM(MACCATALYST)
        return {
            .springMass = 1, 
            .springStiffness = 175, 
            .springDamping = 20, 
            .maximumVelocityMultiplier = 25, 
            .timeToMaximumVelocity = 0.2, 
            .rubberBandForce = 3000 
        };
#else
        return {
            .springMass = 1, 
            .springStiffness = 109, 
            .springDamping = 20, 
            .maximumVelocityMultiplier = 25, 
            .timeToMaximumVelocity = 1.0, 
            .rubberBandForce = 5000 
        };
#endif
    }
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const KeyboardScroll&);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const KeyboardScrollParameters&);

} // namespace WebCore
