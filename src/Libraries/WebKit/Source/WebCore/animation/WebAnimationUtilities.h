/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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

#include "ExceptionOr.h"
#include "RenderStyleConstants.h"
#include "WebAnimationTypes.h"
#include <wtf/Forward.h>
#include <wtf/Markable.h>
#include <wtf/Seconds.h>

namespace WebCore {

enum class PseudoId : uint32_t;

class AnimationEventBase;
class Document;
class Element;
class WebAnimation;

namespace Style {
struct PseudoElementIdentifier;
}

inline double secondsToWebAnimationsAPITime(const Seconds time)
{
    // Precision of time values
    // https://drafts.csswg.org/web-animations-1/#precision-of-time-values

    // The internal representation of time values is implementation dependent however, it is RECOMMENDED that user
    // agents be able to represent input time values with microsecond precision so that a time value (which nominally
    // represents milliseconds) of 0.001 is distinguishable from 0.0.
    auto roundedTime = std::round(time.microseconds()) / 1000;
    if (roundedTime == -0)
        return 0;
    return roundedTime;
}

const auto timeEpsilon = Seconds::fromMilliseconds(0.001);

bool compareAnimationsByCompositeOrder(const WebAnimation&, const WebAnimation&);
bool compareAnimationEventsByCompositeOrder(const AnimationEventBase&, const AnimationEventBase&);
String pseudoElementIdentifierAsString(const std::optional<Style::PseudoElementIdentifier>&);
std::pair<bool, std::optional<Style::PseudoElementIdentifier>> pseudoElementIdentifierFromString(const String&, Document*);
AtomString animatablePropertyAsString(AnimatableCSSProperty);

} // namespace WebCore

