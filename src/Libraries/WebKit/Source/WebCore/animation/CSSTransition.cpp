/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
#include "CSSTransition.h"

#include "Animation.h"
#include "CSSTransitionEvent.h"
#include "DocumentTimeline.h"
#include "InspectorInstrumentation.h"
#include "KeyframeEffect.h"
#include "StyleResolver.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSTransition);

Ref<CSSTransition> CSSTransition::create(const Styleable& owningElement, const AnimatableCSSProperty& property, MonotonicTime generationTime, const Animation& backingAnimation, const RenderStyle& oldStyle, const RenderStyle& newStyle, Seconds delay, Seconds duration, const RenderStyle& reversingAdjustedStartStyle, double reversingShorteningFactor)
{
    auto result = adoptRef(*new CSSTransition(owningElement, property, generationTime, backingAnimation, oldStyle, newStyle, reversingAdjustedStartStyle, reversingShorteningFactor));
    result->initialize(&oldStyle, newStyle, { nullptr });
    result->setTimingProperties(delay, duration);

    InspectorInstrumentation::didCreateWebAnimation(result.get());

    return result;
}

CSSTransition::CSSTransition(const Styleable& styleable, const AnimatableCSSProperty& property, MonotonicTime generationTime, const Animation& backingAnimation, const RenderStyle& oldStyle, const RenderStyle& targetStyle, const RenderStyle& reversingAdjustedStartStyle, double reversingShorteningFactor)
    : StyleOriginatedAnimation(styleable, backingAnimation)
    , m_property(property)
    , m_generationTime(generationTime)
    , m_timelineTimeAtCreation(styleable.element.document().timeline().currentTime())
    , m_targetStyle(RenderStyle::clonePtr(targetStyle))
    , m_currentStyle(RenderStyle::clonePtr(oldStyle))
    , m_reversingAdjustedStartStyle(RenderStyle::clonePtr(reversingAdjustedStartStyle))
    , m_reversingShorteningFactor(reversingShorteningFactor)
{
}

CSSTransition::~CSSTransition() = default;

OptionSet<AnimationImpact> CSSTransition::resolve(RenderStyle& targetStyle, const Style::ResolutionContext& resolutionContext, std::optional<Seconds> startTime)
{
    auto impact = StyleOriginatedAnimation::resolve(targetStyle, resolutionContext, startTime);
    m_currentStyle = RenderStyle::clonePtr(targetStyle);
    return impact;
}

void CSSTransition::animationDidFinish()
{
    StyleOriginatedAnimation::animationDidFinish();

    if (auto owningElement = this->owningElement())
        owningElement->removeStyleOriginatedAnimationFromListsForOwningElement(*this);
}

void CSSTransition::setTimingProperties(Seconds delay, Seconds duration)
{
    suspendEffectInvalidation();

    // This method is only called from CSSTransition::create() where we're guaranteed to have an effect.
    auto* animationEffect = effect();
    ASSERT(animationEffect);

    // In order for CSS Transitions to be seeked backwards, they need to have their fill mode set to backwards
    // such that the original CSS value applied prior to the transition is used for a negative current time.
    animationEffect->setFill(FillMode::Backwards);
    animationEffect->setDelay(delay);
    animationEffect->setIterationDuration(duration);
    animationEffect->setTimingFunction(backingAnimation().timingFunction());
    effectTimingDidChange();

    unsuspendEffectInvalidation();
}

Ref<StyleOriginatedAnimationEvent> CSSTransition::createEvent(const AtomString& eventType, std::optional<Seconds> scheduledTime, double elapsedTime, const std::optional<Style::PseudoElementIdentifier>& pseudoElementIdentifier)
{
    return CSSTransitionEvent::create(eventType, this, scheduledTime, elapsedTime, pseudoElementIdentifier, transitionProperty());
}

const AtomString CSSTransition::transitionProperty() const
{
    return WTF::switchOn(m_property,
        [] (CSSPropertyID cssProperty) {
            return nameString(cssProperty);
        },
        [] (const AtomString& customProperty) {
            return customProperty;
        }
    );
}

} // namespace WebCore
