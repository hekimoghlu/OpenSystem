/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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

#include "AnimationList.h"
#include "AnimationMalloc.h"
#include "CSSPropertyNames.h"
#include "WebAnimationTypes.h"
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
#include <wtf/WeakListHashSet.h>
#endif

namespace WebCore {

class Document;
class KeyframeEffect;
class RenderStyle;
class Settings;

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
class AcceleratedEffect;
#endif

namespace Style {
struct ResolutionContext;
}

class KeyframeEffectStack {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Animation);
public:
    explicit KeyframeEffectStack();
    ~KeyframeEffectStack();

    bool addEffect(KeyframeEffect&);
    void removeEffect(KeyframeEffect&);
    bool hasEffects() const { return !m_effects.isEmpty(); }
    Vector<WeakPtr<KeyframeEffect>> sortedEffects();
    const AnimationList* cssAnimationList() const { return m_cssAnimationList.get(); }
    void setCSSAnimationList(RefPtr<const AnimationList>&&);
    bool containsProperty(CSSPropertyID) const;
    bool isCurrentlyAffectingProperty(CSSPropertyID) const;
    bool requiresPseudoElement() const;
    OptionSet<AnimationImpact> applyKeyframeEffects(RenderStyle& targetStyle, UncheckedKeyHashSet<AnimatableCSSProperty>& affectedProperties, const RenderStyle* previousLastStyleChangeEventStyle, const Style::ResolutionContext&);
    bool hasEffectWithImplicitKeyframes() const;

    void effectAbilityToBeAcceleratedDidChange(const KeyframeEffect&);
    bool allowsAcceleration() const;

    void clearInvalidCSSAnimationNames();
    bool hasInvalidCSSAnimationNames() const;
    bool containsInvalidCSSAnimationName(const String&) const;
    void addInvalidCSSAnimationName(const String&);

    void lastStyleChangeEventStyleDidChange(const RenderStyle* previousStyle, const RenderStyle* currentStyle);
    void cascadeDidOverrideProperties(const UncheckedKeyHashSet<AnimatableCSSProperty>&, const Document&);

    const UncheckedKeyHashSet<AnimatableCSSProperty>& acceleratedPropertiesOverriddenByCascade() const { return m_acceleratedPropertiesOverriddenByCascade; }

    void applyPendingAcceleratedActions() const;

    bool hasAcceleratedEffects(const Settings&) const;
#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    void setAcceleratedEffects(WeakListHashSet<AcceleratedEffect>&& acceleratedEffects) { m_acceleratedEffects = WTFMove(acceleratedEffects); }
#endif

private:
    void ensureEffectsAreSorted();
    bool hasMatchingEffect(const Function<bool(const KeyframeEffect&)>&) const;
    void startAcceleratedAnimationsIfPossible();
    void stopAcceleratedAnimations();

    Vector<WeakPtr<KeyframeEffect>> m_effects;
#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    WeakListHashSet<AcceleratedEffect> m_acceleratedEffects;
#endif
    UncheckedKeyHashSet<String> m_invalidCSSAnimationNames;
    UncheckedKeyHashSet<AnimatableCSSProperty> m_acceleratedPropertiesOverriddenByCascade;
    RefPtr<const AnimationList> m_cssAnimationList;
    bool m_isSorted { true };
};

} // namespace WebCore
