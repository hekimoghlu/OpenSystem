/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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

#include "AnimationEffect.h"
#include "AnimationEffectTiming.h"
#include "BasicEffectTiming.h"
#include "ComputedEffectTiming.h"
#include "ExceptionOr.h"
#include "FillMode.h"
#include "KeyframeEffectOptions.h"
#include "OptionalEffectTiming.h"
#include "PlaybackDirection.h"
#include "TimingFunction.h"
#include "WebAnimation.h"
#include "WebAnimationUtilities.h"
#include <variant>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RefPtr.h>
#include <wtf/Seconds.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class AnimationEffect : public RefCountedAndCanMakeWeakPtr<AnimationEffect> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AnimationEffect);
public:
    virtual ~AnimationEffect();

    virtual bool isCustomEffect() const { return false; }
    virtual bool isKeyframeEffect() const { return false; }

    EffectTiming getBindingsTiming() const;
    BasicEffectTiming getBasicTiming(std::optional<WebAnimationTime> = std::nullopt);
    ComputedEffectTiming getBindingsComputedTiming();
    ComputedEffectTiming getComputedTiming(std::optional<WebAnimationTime> = std::nullopt);
    ExceptionOr<void> bindingsUpdateTiming(Document&, std::optional<OptionalEffectTiming>);
    ExceptionOr<void> updateTiming(Document&, std::optional<OptionalEffectTiming>);

    virtual void animationDidTick() { };
    virtual void animationDidChangeTimingProperties() { };
    virtual void animationWasCanceled() { };
    virtual void animationSuspensionStateDidChange(bool) { };
    virtual void animationTimelineDidChange(const AnimationTimeline*);
    virtual void animationDidFinish() { };
    void animationPlaybackRateDidChange();
    void animationProgressBasedTimelineSourceDidChangeMetrics();
    void animationRangeDidChange();

    AnimationEffectTiming timing() const { return m_timing; }

    WebAnimation* animation() const { return m_animation.get(); }
    virtual void setAnimation(WebAnimation*);

    WebAnimationTime delay();
    Seconds specifiedDelay() const { return m_timing.specifiedStartDelay; }
    void setDelay(const Seconds&);

    WebAnimationTime endDelay();
    Seconds specifiedEndDelay() const { return m_timing.specifiedEndDelay; }
    void setEndDelay(const Seconds&);

    FillMode fill() const { return m_timing.fill; }
    void setFill(FillMode);

    double iterationStart() const { return m_timing.iterationStart; }
    ExceptionOr<void> setIterationStart(double);

    double iterations() const { return m_timing.iterations; }
    ExceptionOr<void> setIterations(double);

    WebAnimationTime iterationDuration();
    std::optional<Seconds> specifiedIterationDuration() const { return m_timing.specifiedIterationDuration; }
    void setIterationDuration(const std::optional<Seconds>&);

    PlaybackDirection direction() const { return m_timing.direction; }
    void setDirection(PlaybackDirection);

    TimingFunction* timingFunction() const { return m_timing.timingFunction.get(); }
    void setTimingFunction(const RefPtr<TimingFunction>&);

    WebAnimationTime activeDuration();
    WebAnimationTime endTime();

    virtual Seconds timeToNextTick(const BasicEffectTiming&);

    virtual bool preventsAnimationReadiness() const { return false; }

protected:
    explicit AnimationEffect();

    virtual bool ticksContinuouslyWhileActive() const { return false; }
    virtual std::optional<double> progressUntilNextStep(double) const;

private:
    AnimationEffectTiming::ResolutionData resolutionData(std::optional<WebAnimationTime>) const;
    void updateComputedTimingPropertiesIfNeeded();

    AnimationEffectTiming m_timing;
    WeakPtr<WebAnimation, WeakPtrImplWithEventTargetData> m_animation;
    bool m_timingDidMutate { false };
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_ANIMATION_EFFECT(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToValueTypeName) \
static bool isType(const WebCore::AnimationEffect& value) { return value.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()
