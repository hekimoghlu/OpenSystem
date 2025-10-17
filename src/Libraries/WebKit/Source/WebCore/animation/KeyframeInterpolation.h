/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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

#include "CompositeOperation.h"
#include "IterationCompositeOperation.h"
#include "TimingFunction.h"
#include "WebAnimationTypes.h"
#include <optional>
#include <wtf/Seconds.h>

namespace WebCore {

class KeyframeInterpolation {
public:
    using Property = std::variant<AnimatableCSSProperty, AcceleratedEffectProperty>;

    class Keyframe {
    public:
        virtual double offset() const = 0;
        virtual std::optional<CompositeOperation> compositeOperation() const = 0;
        virtual bool animatesProperty(Property) const = 0;

        virtual bool isAcceleratedEffectKeyframe() const { return false; }
        virtual bool isBlendingKeyframe() const { return false; }

        virtual ~Keyframe() = default;
    };

    virtual CompositeOperation compositeOperation() const = 0;
    virtual bool isPropertyAdditiveOrCumulative(Property) const = 0;
    virtual IterationCompositeOperation iterationCompositeOperation() const { return IterationCompositeOperation::Replace; }
    virtual const Keyframe& keyframeAtIndex(size_t) const = 0;
    virtual size_t numberOfKeyframes() const = 0;
    virtual const TimingFunction* timingFunctionForKeyframe(const Keyframe&) const = 0;

    struct KeyframeInterval {
        const Vector<const Keyframe*> endpoints;
        bool hasImplicitZeroKeyframe { false };
        bool hasImplicitOneKeyframe { false };
    };

    const KeyframeInterval interpolationKeyframes(Property, double iterationProgress, const Keyframe& defaultStartKeyframe, const Keyframe& defaultEndKeyframe) const;

    using CompositionCallback = Function<void(const Keyframe&, CompositeOperation)>;
    using AccumulationCallback = Function<void(const Keyframe&)>;
    using InterpolationCallback = Function<void(double intervalProgress, double currentIteration, IterationCompositeOperation)>;
    using RequiresBlendingForAccumulativeIterationCallback = Function<bool()>;
    void interpolateKeyframes(Property, const KeyframeInterval&, double iterationProgress, double currentIteration, const WebAnimationTime& iterationDuration, TimingFunction::Before, const CompositionCallback&, const AccumulationCallback&, const InterpolationCallback&, const RequiresBlendingForAccumulativeIterationCallback&) const;

    virtual ~KeyframeInterpolation() = default;
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_KEYFRAME_INTERPOLATION_KEYFRAME(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToValueTypeName) \
static bool isType(const WebCore::KeyframeInterpolation::Keyframe& value) { return value.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()
