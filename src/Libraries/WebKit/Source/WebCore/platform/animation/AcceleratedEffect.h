/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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

#if ENABLE(THREADED_ANIMATION_RESOLUTION)

#include "AcceleratedEffectValues.h"
#include "AnimationEffectTiming.h"
#include "CompositeOperation.h"
#include "KeyframeInterpolation.h"
#include "TimingFunction.h"
#include "WebAnimationTypes.h"
#include <wtf/OptionSet.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class FloatRect;
class IntRect;
class KeyframeEffect;

class AcceleratedEffect : public RefCountedAndCanMakeWeakPtr<AcceleratedEffect>, public KeyframeInterpolation {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AcceleratedEffect);
public:

    class WEBCORE_EXPORT Keyframe final : public KeyframeInterpolation::Keyframe {
    public:
        Keyframe(double offset, AcceleratedEffectValues&&);
        Keyframe(double offset, AcceleratedEffectValues&&, RefPtr<TimingFunction>&&, std::optional<CompositeOperation>, OptionSet<AcceleratedEffectProperty>&&);
        Keyframe clone() const;

        // KeyframeInterpolation::Keyframe
        double offset() const final { return m_offset; }
        std::optional<CompositeOperation> compositeOperation() const final { return m_compositeOperation; }
        bool animatesProperty(KeyframeInterpolation::Property) const final;
        bool isAcceleratedEffectKeyframe() const final { return true; }

        void clearProperty(AcceleratedEffectProperty);
        const OptionSet<AcceleratedEffectProperty>& animatedProperties() const { return m_animatedProperties; }
        const RefPtr<TimingFunction>& timingFunction() const { return m_timingFunction; }
        const AcceleratedEffectValues& values() const { return m_values; }

    private:
        double m_offset;
        AcceleratedEffectValues m_values;
        RefPtr<TimingFunction> m_timingFunction;
        std::optional<CompositeOperation> m_compositeOperation;
        OptionSet<AcceleratedEffectProperty> m_animatedProperties;
    };

    static RefPtr<AcceleratedEffect> create(const KeyframeEffect&, const IntRect&, const AcceleratedEffectValues&, OptionSet<AcceleratedEffectProperty>&);
    WEBCORE_EXPORT static Ref<AcceleratedEffect> create(AnimationEffectTiming, Vector<Keyframe>&&, WebAnimationType, CompositeOperation, RefPtr<TimingFunction>&& defaultKeyframeTimingFunction, OptionSet<AcceleratedEffectProperty>&&, bool paused, double playbackRate, std::optional<WebAnimationTime> startTime, std::optional<WebAnimationTime> holdTime);

    virtual ~AcceleratedEffect() = default;

    WEBCORE_EXPORT Ref<AcceleratedEffect> clone() const;
    WEBCORE_EXPORT Ref<AcceleratedEffect> copyWithProperties(OptionSet<AcceleratedEffectProperty>&) const;

    WEBCORE_EXPORT void apply(WebAnimationTime, AcceleratedEffectValues&, const FloatRect&);

    // Encoding and decoding support
    AnimationEffectTiming timing() const { return m_timing; }
    const Vector<Keyframe>& keyframes() const { return m_keyframes; }
    WebAnimationType animationType() const { return m_animationType; }
    CompositeOperation compositeOperation() const final { return m_compositeOperation; }
    const RefPtr<TimingFunction>& defaultKeyframeTimingFunction() const { return m_defaultKeyframeTimingFunction; }
    const OptionSet<AcceleratedEffectProperty>& animatedProperties() const { return m_animatedProperties; }
    bool paused() const { return m_paused; }
    double playbackRate() const { return m_playbackRate; }
    std::optional<WebAnimationTime> startTime() const { return m_startTime; }
    std::optional<WebAnimationTime> holdTime() const { return m_holdTime; }

    const OptionSet<AcceleratedEffectProperty>& disallowedProperties() const { return m_disallowedProperties; }

    bool animatesTransformRelatedProperty() const;

private:
    AcceleratedEffect(const KeyframeEffect&, const IntRect&, const OptionSet<AcceleratedEffectProperty>&);
    explicit AcceleratedEffect(AnimationEffectTiming, Vector<Keyframe>&&, WebAnimationType, CompositeOperation, RefPtr<TimingFunction>&& defaultKeyframeTimingFunction, OptionSet<AcceleratedEffectProperty>&&, bool paused, double playbackRate, std::optional<WebAnimationTime> startTime, std::optional<WebAnimationTime> holdTime);
    explicit AcceleratedEffect(const AcceleratedEffect&, OptionSet<AcceleratedEffectProperty>&);

    void validateFilters(const AcceleratedEffectValues& baseValues, OptionSet<AcceleratedEffectProperty>&);

    // KeyframeInterpolation
    bool isPropertyAdditiveOrCumulative(KeyframeInterpolation::Property) const final;
    const KeyframeInterpolation::Keyframe& keyframeAtIndex(size_t) const final;
    size_t numberOfKeyframes() const final { return m_keyframes.size(); }
    const TimingFunction* timingFunctionForKeyframe(const KeyframeInterpolation::Keyframe&) const final;

    AnimationEffectTiming m_timing;
    Vector<Keyframe> m_keyframes;
    WebAnimationType m_animationType { WebAnimationType::WebAnimation };
    CompositeOperation m_compositeOperation { CompositeOperation::Replace };
    RefPtr<TimingFunction> m_defaultKeyframeTimingFunction;
    OptionSet<AcceleratedEffectProperty> m_animatedProperties;
    OptionSet<AcceleratedEffectProperty> m_disallowedProperties;
    bool m_paused { false };
    double m_playbackRate { 1 };
    std::optional<WebAnimationTime> m_startTime;
    std::optional<WebAnimationTime> m_holdTime;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_KEYFRAME_INTERPOLATION_KEYFRAME(AcceleratedEffect::Keyframe, isAcceleratedEffectKeyframe());

#endif // ENABLE(THREADED_ANIMATION_RESOLUTION)
