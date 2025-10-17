/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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
#include "CSSAnimation.h"

#include "AnimationEffect.h"
#include "AnimationTimelinesController.h"
#include "CSSAnimationEvent.h"
#include "DocumentTimeline.h"
#include "InspectorInstrumentation.h"
#include "KeyframeEffect.h"
#include "RenderStyle.h"
#include "ViewTimeline.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSAnimation);

Ref<CSSAnimation> CSSAnimation::create(const Styleable& owningElement, const Animation& backingAnimation, const RenderStyle* oldStyle, const RenderStyle& newStyle, const Style::ResolutionContext& resolutionContext)
{
    auto result = adoptRef(*new CSSAnimation(owningElement, backingAnimation));
    result->initialize(oldStyle, newStyle, resolutionContext);

    InspectorInstrumentation::didCreateWebAnimation(result.get());

    return result;
}

CSSAnimation::CSSAnimation(const Styleable& element, const Animation& backingAnimation)
    : StyleOriginatedAnimation(element, backingAnimation)
    , m_animationName(backingAnimation.name().name)
{
}

void CSSAnimation::syncPropertiesWithBackingAnimation()
{
    StyleOriginatedAnimation::syncPropertiesWithBackingAnimation();

    // If we have been disassociated from our original owning element,
    // we should no longer sync any of the `animation-*` CSS properties.
    if (!owningElement())
        return;

    if (!effect())
        return;

    suspendEffectInvalidation();

    auto& animation = backingAnimation();
    auto* animationEffect = effect();

    if (!m_overriddenProperties.contains(Property::FillMode)) {
        switch (animation.fillMode()) {
        case AnimationFillMode::None:
            animationEffect->setFill(FillMode::None);
            break;
        case AnimationFillMode::Backwards:
            animationEffect->setFill(FillMode::Backwards);
            break;
        case AnimationFillMode::Forwards:
            animationEffect->setFill(FillMode::Forwards);
            break;
        case AnimationFillMode::Both:
            animationEffect->setFill(FillMode::Both);
            break;
        }
    }

    if (!m_overriddenProperties.contains(Property::Direction)) {
        switch (animation.direction()) {
        case Animation::Direction::Normal:
            animationEffect->setDirection(PlaybackDirection::Normal);
            break;
        case Animation::Direction::Alternate:
            animationEffect->setDirection(PlaybackDirection::Alternate);
            break;
        case Animation::Direction::Reverse:
            animationEffect->setDirection(PlaybackDirection::Reverse);
            break;
        case Animation::Direction::AlternateReverse:
            animationEffect->setDirection(PlaybackDirection::AlternateReverse);
            break;
        }
    }

    if (!m_overriddenProperties.contains(Property::IterationCount)) {
        auto iterationCount = animation.iterationCount();
        animationEffect->setIterations(iterationCount == Animation::IterationCountInfinite ? std::numeric_limits<double>::infinity() : iterationCount);
    }

    if (!m_overriddenProperties.contains(Property::Delay))
        animationEffect->setDelay(Seconds(animation.delay()));

    if (!m_overriddenProperties.contains(Property::Duration)) {
        if (auto duration = animation.duration())
            animationEffect->setIterationDuration(Seconds(*duration));
        else
            animationEffect->setIterationDuration(std::nullopt);
    }

    if (!m_overriddenProperties.contains(Property::CompositeOperation)) {
        if (auto* keyframeEffect = dynamicDowncast<KeyframeEffect>(animationEffect))
            keyframeEffect->setComposite(animation.compositeOperation());
    }

    syncStyleOriginatedTimeline();

    if (!m_overriddenProperties.contains(Property::RangeStart))
        setRangeStart(animation.range().start);
    if (!m_overriddenProperties.contains(Property::RangeEnd))
        setRangeEnd(animation.range().end);

    effectTimingDidChange();

    // Synchronize the play state
    if (!m_overriddenProperties.contains(Property::PlayState)) {
        if (animation.playState() == AnimationPlayState::Playing && playState() == WebAnimation::PlayState::Paused)
            play();
        else if (animation.playState() == AnimationPlayState::Paused && playState() == WebAnimation::PlayState::Running)
            pause();
    }

    unsuspendEffectInvalidation();
}

void CSSAnimation::syncStyleOriginatedTimeline()
{
    if (m_overriddenProperties.contains(Property::Timeline) || !effect())
        return;

    suspendEffectInvalidation();

    ASSERT(owningElement());
    Ref target = owningElement()->element;
    Ref document = owningElement()->element.document();
    WTF::switchOn(backingAnimation().timeline(),
        [&] (Animation::TimelineKeyword keyword) {
            setTimeline(keyword == Animation::TimelineKeyword::None ? nullptr : RefPtr { document->existingTimeline() });
        }, [&] (const AtomString& name) {
            CheckedRef timelinesController = document->ensureTimelinesController();
            timelinesController->setTimelineForName(name, target, *this);
        }, [&] (const Animation::AnonymousScrollTimeline& anonymousScrollTimeline) {
            auto scrollTimeline = ScrollTimeline::create(anonymousScrollTimeline.scroller, anonymousScrollTimeline.axis);
            if (auto owningElement = this->owningElement())
                scrollTimeline->setSource(*owningElement);
            else
                scrollTimeline->setSource(nullptr);
            setTimeline(WTFMove(scrollTimeline));
        }, [&] (const Animation::AnonymousViewTimeline& anonymousViewTimeline) {
            auto insets = anonymousViewTimeline.insets;
            auto viewTimeline = ViewTimeline::create(nullAtom(), anonymousViewTimeline.axis, WTFMove(insets));
            viewTimeline->setSubject(target.ptr());
            setTimeline(WTFMove(viewTimeline));
        }
    );

    unsuspendEffectInvalidation();
}

AnimationTimeline* CSSAnimation::bindingsTimeline() const
{
    flushPendingStyleChanges();
    return StyleOriginatedAnimation::bindingsTimeline();
}

void CSSAnimation::setBindingsTimeline(RefPtr<AnimationTimeline>&& timeline)
{
    m_overriddenProperties.add(Property::Timeline);
    StyleOriginatedAnimation::setBindingsTimeline(WTFMove(timeline));
}

void CSSAnimation::setBindingsRangeStart(TimelineRangeValue&& range)
{
    m_overriddenProperties.add(Property::RangeStart);
    StyleOriginatedAnimation::setBindingsRangeStart(WTFMove(range));
}

void CSSAnimation::setBindingsRangeEnd(TimelineRangeValue&& range)
{
    m_overriddenProperties.add(Property::RangeEnd);
    StyleOriginatedAnimation::setBindingsRangeEnd(WTFMove(range));
}

ExceptionOr<void> CSSAnimation::bindingsPlay()
{
    // https://drafts.csswg.org/css-animations-2/#animations

    // After a successful call to play() or pause() on a CSSAnimation, any subsequent change to the animation-play-state will
    // no longer cause the CSSAnimation to be played or paused.

    auto retVal = StyleOriginatedAnimation::bindingsPlay();
    if (!retVal.hasException())
        m_overriddenProperties.add(Property::PlayState);
    return retVal;
}

ExceptionOr<void> CSSAnimation::bindingsPause()
{
    // https://drafts.csswg.org/css-animations-2/#animations

    // After a successful call to play() or pause() on a CSSAnimation, any subsequent change to the animation-play-state will
    // no longer cause the CSSAnimation to be played or paused.

    auto retVal = StyleOriginatedAnimation::bindingsPause();
    if (!retVal.hasException())
        m_overriddenProperties.add(Property::PlayState);
    return retVal;
}

void CSSAnimation::setBindingsEffect(RefPtr<AnimationEffect>&& newEffect)
{
    // https://drafts.csswg.org/css-animations-2/#animations

    // After successfully setting the effect of a CSSAnimation to null or some AnimationEffect other than the original KeyframeEffect,
    // all subsequent changes to animation properties other than animation-name or animation-play-state will not be reflected in that
    // animation. Similarly, any change to matching @keyframes rules will not be reflected in that animation. However, if the last
    // matching @keyframes rule is removed the animation must still be canceled.

    auto* previousEffect = effect();
    StyleOriginatedAnimation::setBindingsEffect(WTFMove(newEffect));
    if (effect() != previousEffect) {
        m_overriddenProperties.add(Property::Duration);
        m_overriddenProperties.add(Property::TimingFunction);
        m_overriddenProperties.add(Property::IterationCount);
        m_overriddenProperties.add(Property::Direction);
        m_overriddenProperties.add(Property::Delay);
        m_overriddenProperties.add(Property::FillMode);
        m_overriddenProperties.add(Property::Keyframes);
        m_overriddenProperties.add(Property::CompositeOperation);
    }
}

ExceptionOr<void> CSSAnimation::setBindingsStartTime(const std::optional<WebAnimationTime>& startTime)
{
    // https://drafts.csswg.org/css-animations-2/#animations

    // After a successful call to reverse() on a CSSAnimation or after successfully setting the startTime on a CSSAnimation,
    // if, as a result of that call the play state of the CSSAnimation changes to or from the paused play state, any subsequent
    // change to the animation-play-state will no longer cause the CSSAnimation to be played or paused.

    auto previousPlayState = playState();
    auto result = StyleOriginatedAnimation::setBindingsStartTime(startTime);
    if (result.hasException())
        return result.releaseException();
    auto currentPlayState = playState();
    if (currentPlayState != previousPlayState && (currentPlayState == PlayState::Paused || previousPlayState == PlayState::Paused))
        m_overriddenProperties.add(Property::PlayState);

    return { };
}

ExceptionOr<void> CSSAnimation::bindingsReverse()
{
    // https://drafts.csswg.org/css-animations-2/#animations

    // After a successful call to reverse() on a CSSAnimation or after successfully setting the startTime on a CSSAnimation,
    // if, as a result of that call the play state of the CSSAnimation changes to or from the paused play state, any subsequent
    // change to the animation-play-state will no longer cause the CSSAnimation to be played or paused.

    auto previousPlayState = playState();
    auto retVal = StyleOriginatedAnimation::bindingsReverse();
    if (!retVal.hasException()) {
        auto currentPlayState = playState();
        if (currentPlayState != previousPlayState && (currentPlayState == PlayState::Paused || previousPlayState == PlayState::Paused))
            m_overriddenProperties.add(Property::PlayState);
    }
    return retVal;
}

void CSSAnimation::effectTimingWasUpdatedUsingBindings(OptionalEffectTiming timing)
{
    // https://drafts.csswg.org/css-animations-2/#animations

    // After a successful call to updateTiming() on the KeyframeEffect associated with a CSSAnimation, for each property
    // included in the timing parameter, any subsequent change to a corresponding animation property will not be reflected
    // in that animation.

    if (timing.duration)
        m_overriddenProperties.add(Property::Duration);

    if (timing.iterations)
        m_overriddenProperties.add(Property::IterationCount);

    if (timing.delay)
        m_overriddenProperties.add(Property::Delay);

    if (!timing.easing.isNull())
        m_overriddenProperties.add(Property::TimingFunction);

    if (timing.fill)
        m_overriddenProperties.add(Property::FillMode);

    if (timing.direction)
        m_overriddenProperties.add(Property::Direction);
}

void CSSAnimation::effectKeyframesWereSetUsingBindings()
{
    // https://drafts.csswg.org/css-animations-2/#animations

    // After a successful call to setKeyframes() on the KeyframeEffect associated with a CSSAnimation, any subsequent change to
    // matching @keyframes rules or the resolved value of the animation-timing-function property for the target element will not
    // be reflected in that animation.
    m_overriddenProperties.add(Property::Keyframes);
    m_overriddenProperties.add(Property::TimingFunction);
}

void CSSAnimation::effectCompositeOperationWasSetUsingBindings()
{
    m_overriddenProperties.add(Property::CompositeOperation);
}

void CSSAnimation::keyframesRuleDidChange()
{
    if (m_overriddenProperties.contains(Property::Keyframes))
        return;

    auto* keyframeEffect = dynamicDowncast<KeyframeEffect>(effect());
    if (!keyframeEffect)
        return;

    auto owningElement = this->owningElement();
    if (!owningElement)
        return;

    keyframeEffect->keyframesRuleDidChange();
    owningElement->keyframesRuleDidChange();
}

void CSSAnimation::updateKeyframesIfNeeded(const RenderStyle* oldStyle, const RenderStyle& newStyle, const Style::ResolutionContext& resolutionContext)
{
    if (m_overriddenProperties.contains(Property::Keyframes))
        return;

    auto* keyframeEffect = dynamicDowncast<KeyframeEffect>(effect());
    if (!keyframeEffect)
        return;

    if (keyframeEffect->blendingKeyframes().isEmpty())
        keyframeEffect->computeStyleOriginatedAnimationBlendingKeyframes(oldStyle, newStyle, resolutionContext);
}

Ref<StyleOriginatedAnimationEvent> CSSAnimation::createEvent(const AtomString& eventType, std::optional<Seconds> scheduledTime, double elapsedTime, const std::optional<Style::PseudoElementIdentifier>& pseudoElementIdentifier)
{
    return CSSAnimationEvent::create(eventType, this, scheduledTime, elapsedTime, pseudoElementIdentifier, m_animationName);
}

} // namespace WebCore
