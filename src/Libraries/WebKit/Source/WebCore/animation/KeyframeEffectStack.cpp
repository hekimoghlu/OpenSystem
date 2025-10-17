/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
#include "KeyframeEffectStack.h"

#include "AnimationTimeline.h"
#include "CSSAnimation.h"
#include "CSSPropertyAnimation.h"
#include "CSSTransition.h"
#include "Document.h"
#include "KeyframeEffect.h"
#include "RenderStyleInlines.h"
#include "RotateTransformOperation.h"
#include "ScaleTransformOperation.h"
#include "Settings.h"
#include "TransformOperations.h"
#include "TranslateTransformOperation.h"
#include "WebAnimation.h"
#include "WebAnimationUtilities.h"
#include <wtf/PointerComparison.h>

namespace WebCore {

KeyframeEffectStack::KeyframeEffectStack() = default;

KeyframeEffectStack::~KeyframeEffectStack() = default;

bool KeyframeEffectStack::addEffect(KeyframeEffect& effect)
{
    // To qualify for membership in an effect stack, an effect must have a target, an animation, a timeline and be relevant.
    // This method will be called in WebAnimation and KeyframeEffect as those properties change.
    if (!effect.targetStyleable() || !effect.animation() || !effect.animation()->timeline() || !effect.animation()->isRelevant())
        return false;

    m_effects.append(effect);
    m_isSorted = false;

    if (m_effects.size() > 1 && effect.preventsAcceleration())
        stopAcceleratedAnimations();

    effect.wasAddedToEffectStack();

    return true;
}

void KeyframeEffectStack::removeEffect(KeyframeEffect& effect)
{
    auto removedEffect = m_effects.removeFirst(&effect);

    if (removedEffect)
        effect.wasRemovedFromEffectStack();

    if (!removedEffect || m_effects.isEmpty())
        return;

    if (!effect.canBeAccelerated())
        startAcceleratedAnimationsIfPossible();
}

bool KeyframeEffectStack::hasMatchingEffect(const Function<bool(const KeyframeEffect&)>& function) const
{
    for (auto& effect : m_effects) {
        if (function(*effect))
            return true;
    }
    return false;
}

bool KeyframeEffectStack::containsProperty(CSSPropertyID property) const
{
    return hasMatchingEffect([property] (const KeyframeEffect& effect) {
        return effect.animatesProperty(property);
    });
}

bool KeyframeEffectStack::requiresPseudoElement() const
{
    return hasMatchingEffect([] (const KeyframeEffect& effect) {
        return effect.requiresPseudoElement();
    });
}

bool KeyframeEffectStack::hasEffectWithImplicitKeyframes() const
{
    return hasMatchingEffect([] (const KeyframeEffect& effect) {
        return effect.hasImplicitKeyframes();
    });
}

bool KeyframeEffectStack::isCurrentlyAffectingProperty(CSSPropertyID property) const
{
    return hasMatchingEffect([property] (const KeyframeEffect& effect) {
        return effect.isCurrentlyAffectingProperty(property) || effect.isRunningAcceleratedAnimationForProperty(property);
    });
}

Vector<WeakPtr<KeyframeEffect>> KeyframeEffectStack::sortedEffects()
{
    ensureEffectsAreSorted();
    return m_effects;
}

void KeyframeEffectStack::ensureEffectsAreSorted()
{
    if (m_isSorted || m_effects.size() < 2)
        return;

    std::stable_sort(m_effects.begin(), m_effects.end(), [](auto& lhs, auto& rhs) {
        RELEASE_ASSERT(lhs.get());
        RELEASE_ASSERT(rhs.get());
        
        auto* lhsAnimation = lhs->animation();
        auto* rhsAnimation = rhs->animation();

        RELEASE_ASSERT(lhsAnimation);
        RELEASE_ASSERT(rhsAnimation);

        return compareAnimationsByCompositeOrder(*lhsAnimation, *rhsAnimation);
    });

    m_isSorted = true;
}

void KeyframeEffectStack::setCSSAnimationList(RefPtr<const AnimationList>&& cssAnimationList)
{
    m_cssAnimationList = WTFMove(cssAnimationList);
    // Since the list of animation names has changed, the sorting order of the animation effects may have changed as well.
    m_isSorted = false;
}

OptionSet<AnimationImpact> KeyframeEffectStack::applyKeyframeEffects(RenderStyle& targetStyle, UncheckedKeyHashSet<AnimatableCSSProperty>& affectedProperties, const RenderStyle* previousLastStyleChangeEventStyle, const Style::ResolutionContext& resolutionContext)
{
    OptionSet<AnimationImpact> impact;

    auto& previousStyle = previousLastStyleChangeEventStyle ? *previousLastStyleChangeEventStyle : RenderStyle::defaultStyle();

    auto transformRelatedPropertyChanged = [&]() -> bool {
        return !arePointingToEqualData(targetStyle.translate(), previousStyle.translate())
            || !arePointingToEqualData(targetStyle.scale(), previousStyle.scale())
            || !arePointingToEqualData(targetStyle.rotate(), previousStyle.rotate())
            || targetStyle.transform() != previousStyle.transform();
    }();

    auto unanimatedStyle = RenderStyle::clone(targetStyle);

    for (const auto& effect : sortedEffects()) {
        auto keyframeRecomputationReason = effect->recomputeKeyframesIfNecessary(previousLastStyleChangeEventStyle, unanimatedStyle, resolutionContext);

        ASSERT(effect->animation());
        auto* animation = effect->animation();
        impact.add(animation->resolve(targetStyle, resolutionContext));

        if (effect->isRunningAccelerated() || effect->isAboutToRunAccelerated())
            impact.add(AnimationImpact::RequiresRecomposite);

        if (effect->triggersStackingContext())
            impact.add(AnimationImpact::ForcesStackingContext);

        if (transformRelatedPropertyChanged && effect->isRunningAcceleratedTransformRelatedAnimation())
            effect->transformRelatedPropertyDidChange();

        // If one of the effect's resolved property changed it could affect whether that effect's animation is removed.
        if (keyframeRecomputationReason && *keyframeRecomputationReason == KeyframeEffect::RecomputationReason::LogicalPropertyChange) {
            ASSERT(animation->timeline());
            animation->timeline()->animationTimingDidChange(*animation);
        }

        affectedProperties.formUnion(effect->animatedProperties());
    }

    return impact;
}

void KeyframeEffectStack::clearInvalidCSSAnimationNames()
{
    m_invalidCSSAnimationNames.clear();
}

bool KeyframeEffectStack::hasInvalidCSSAnimationNames() const
{
    return !m_invalidCSSAnimationNames.isEmpty();
}

bool KeyframeEffectStack::containsInvalidCSSAnimationName(const String& name) const
{
    return m_invalidCSSAnimationNames.contains(name);
}

void KeyframeEffectStack::addInvalidCSSAnimationName(const String& name)
{
    m_invalidCSSAnimationNames.add(name);
}

void KeyframeEffectStack::effectAbilityToBeAcceleratedDidChange(const KeyframeEffect& effect)
{
    ASSERT(m_effects.contains(&effect));
    if (effect.preventsAcceleration())
        stopAcceleratedAnimations();
    else
        startAcceleratedAnimationsIfPossible();
}

bool KeyframeEffectStack::allowsAcceleration() const
{
    // We could try and be a lot smarter here and do this on a per-property basis and
    // account for fully replacing effects which could co-exist with effects that
    // don't support acceleration lower in the stack, etc. But, if we are not able to run
    // all effects that could support acceleration using acceleration, then we might
    // as well not run any at all since we'll be updating effects for this stack
    // for each animation frame. So for now, we simply return false if any effect in the
    // stack is unable to be accelerated, or if we have more than one effect animating
    // an accelerated property with an implicit keyframe.

    UncheckedKeyHashSet<AnimatableCSSProperty> allAcceleratedProperties;

    for (auto& effect : m_effects) {
        if (effect->preventsAcceleration())
            return false;
        auto& acceleratedProperties = effect->acceleratedProperties();
        if (!allAcceleratedProperties.isEmpty()) {
            auto previouslySeenAcceleratedPropertiesAffectingCurrentEffect = allAcceleratedProperties.intersectionWith(acceleratedProperties);
            if (!previouslySeenAcceleratedPropertiesAffectingCurrentEffect.isEmpty()
                && !effect->acceleratedPropertiesWithImplicitKeyframe().intersectionWith(previouslySeenAcceleratedPropertiesAffectingCurrentEffect).isEmpty()) {
                return false;
            }
        }
        allAcceleratedProperties.add(acceleratedProperties.begin(), acceleratedProperties.end());
    }

    return true;
}

void KeyframeEffectStack::startAcceleratedAnimationsIfPossible()
{
    if (!allowsAcceleration())
        return;

    for (auto& effect : m_effects)
        effect->effectStackNoLongerPreventsAcceleration();
}

void KeyframeEffectStack::stopAcceleratedAnimations()
{
    for (auto& effect : m_effects)
        effect->effectStackNoLongerAllowsAcceleration();
}

void KeyframeEffectStack::lastStyleChangeEventStyleDidChange(const RenderStyle* previousStyle, const RenderStyle* currentStyle)
{
    for (auto& effect : m_effects)
        effect->lastStyleChangeEventStyleDidChange(previousStyle, currentStyle);
}

void KeyframeEffectStack::cascadeDidOverrideProperties(const UncheckedKeyHashSet<AnimatableCSSProperty>& overriddenProperties, const Document& document)
{
    UncheckedKeyHashSet<AnimatableCSSProperty> acceleratedPropertiesOverriddenByCascade;
    for (auto animatedProperty : overriddenProperties) {
        if (CSSPropertyAnimation::animationOfPropertyIsAccelerated(animatedProperty, document.settings()))
            acceleratedPropertiesOverriddenByCascade.add(animatedProperty);
    }

    if (acceleratedPropertiesOverriddenByCascade == m_acceleratedPropertiesOverriddenByCascade)
        return;

    m_acceleratedPropertiesOverriddenByCascade = WTFMove(acceleratedPropertiesOverriddenByCascade);

    for (auto& effect : m_effects)
        effect->acceleratedPropertiesOverriddenByCascadeDidChange();
}

void KeyframeEffectStack::applyPendingAcceleratedActions() const
{
    bool hasActiveAcceleratedEffect = m_effects.containsIf([](const auto& effect) {
        return effect->canBeAccelerated() && effect->animation()->playState() == WebAnimation::PlayState::Running;
    });

    auto accelerationWasPrevented = false;

    for (auto& effect : m_effects) {
        if (hasActiveAcceleratedEffect)
            effect->applyPendingAcceleratedActionsOrUpdateTimingProperties();
        else
            effect->applyPendingAcceleratedActions();
        accelerationWasPrevented = accelerationWasPrevented || effect->accelerationWasPrevented() || effect->preventsAcceleration();
    }

    if (accelerationWasPrevented) {
        for (auto& effect : m_effects)
            effect->effectStackNoLongerAllowsAccelerationDuringAcceleratedActionApplication();
    }
}

bool KeyframeEffectStack::hasAcceleratedEffects(const Settings& settings) const
{
#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    if (settings.threadedAnimationResolutionEnabled())
        return !m_acceleratedEffects.isEmptyIgnoringNullReferences();
#else
    UNUSED_PARAM(settings);
#endif
    for (auto& effect : m_effects) {
        if (effect->isRunningAccelerated())
            return true;
    }
    return false;
}

} // namespace WebCore
