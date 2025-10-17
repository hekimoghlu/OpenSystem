/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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

#include "Element.h"
#include "PseudoElement.h"
#include "PseudoElementIdentifier.h"
#include "RenderStyleConstants.h"
#include "WebAnimationTypes.h"

namespace WebCore {

class KeyframeEffectStack;
class RenderElement;
class RenderStyle;
class WebAnimation;

namespace Style {
enum class IsInDisplayNoneTree : bool;
}

struct Styleable {
    Element& element;
    std::optional<Style::PseudoElementIdentifier> pseudoElementIdentifier;

    Styleable(Element& element, const std::optional<Style::PseudoElementIdentifier>& pseudoElementIdentifier)
        : element(element)
        , pseudoElementIdentifier(pseudoElementIdentifier)
    {
    }

    static const Styleable fromElement(Element& element)
    {
        if (auto* pseudoElement = dynamicDowncast<PseudoElement>(element))
            return Styleable(*pseudoElement->hostElement(), Style::PseudoElementIdentifier { element.pseudoId() });
        ASSERT(element.pseudoId() == PseudoId::None);
        return Styleable(element, std::nullopt);
    }

    static const std::optional<const Styleable> fromRenderer(const RenderElement&);

    bool operator==(const Styleable& other) const
    {
        return (&element == &other.element && pseudoElementIdentifier == other.pseudoElementIdentifier);
    }

    RenderElement* renderer() const;

    std::unique_ptr<RenderStyle> computeAnimatedStyle() const;

    // If possible, compute the visual extent of any transform animation using the given rect,
    // returning the result in the rect. Return false if there is some transform animation but
    // we were unable to cheaply compute its effect on the extent.
    bool computeAnimationExtent(LayoutRect&) const;

    bool mayHaveNonZeroOpacity() const;

    bool isRunningAcceleratedTransformAnimation() const;

    bool hasRunningAcceleratedAnimations() const;

    bool capturedInViewTransition() const;
    void setCapturedInViewTransition(AtomString);

    KeyframeEffectStack* keyframeEffectStack() const
    {
        return element.keyframeEffectStack(pseudoElementIdentifier);
    }

    KeyframeEffectStack& ensureKeyframeEffectStack() const
    {
        return element.ensureKeyframeEffectStack(pseudoElementIdentifier);
    }

    bool hasKeyframeEffects() const
    {
        return element.hasKeyframeEffects(pseudoElementIdentifier);
    }

    OptionSet<AnimationImpact> applyKeyframeEffects(RenderStyle& targetStyle, UncheckedKeyHashSet<AnimatableCSSProperty>& affectedProperties, const RenderStyle* previousLastStyleChangeEventStyle, const Style::ResolutionContext&) const;

    const AnimationCollection* animations() const
    {
        return element.animations(pseudoElementIdentifier);
    }

    bool hasCompletedTransitionForProperty(const AnimatableCSSProperty& property) const
    {
        return element.hasCompletedTransitionForProperty(pseudoElementIdentifier, property);
    }

    bool hasRunningTransitionForProperty(const AnimatableCSSProperty& property) const
    {
        return element.hasRunningTransitionForProperty(pseudoElementIdentifier, property);
    }

    bool hasRunningTransitions() const
    {
        return element.hasRunningTransitions(pseudoElementIdentifier);
    }

    AnimationCollection& ensureAnimations() const
    {
        return element.ensureAnimations(pseudoElementIdentifier);
    }

    AnimatableCSSPropertyToTransitionMap& ensureCompletedTransitionsByProperty() const
    {
        return element.ensureCompletedTransitionsByProperty(pseudoElementIdentifier);
    }

    AnimatableCSSPropertyToTransitionMap& ensureRunningTransitionsByProperty() const
    {
        return element.ensureRunningTransitionsByProperty(pseudoElementIdentifier);
    }

    CSSAnimationCollection& animationsCreatedByMarkup() const
    {
        return element.animationsCreatedByMarkup(pseudoElementIdentifier);
    }

    void setAnimationsCreatedByMarkup(CSSAnimationCollection&& collection) const
    {
        element.setAnimationsCreatedByMarkup(pseudoElementIdentifier, WTFMove(collection));
    }

    const RenderStyle* lastStyleChangeEventStyle() const
    {
        return element.lastStyleChangeEventStyle(pseudoElementIdentifier);
    }

    void setLastStyleChangeEventStyle(std::unique_ptr<const RenderStyle>&& style) const
    {
        element.setLastStyleChangeEventStyle(pseudoElementIdentifier, WTFMove(style));
    }

    bool hasPropertiesOverridenAfterAnimation() const
    {
        return element.hasPropertiesOverridenAfterAnimation(pseudoElementIdentifier);
    }

    void setHasPropertiesOverridenAfterAnimation(bool value) const
    {
        element.setHasPropertiesOverridenAfterAnimation(pseudoElementIdentifier, value);
    }

    void keyframesRuleDidChange() const
    {
        element.keyframesRuleDidChange(pseudoElementIdentifier);
    }

    void queryContainerDidChange() const;

    bool animationListContainsNewlyValidAnimation(const AnimationList&) const;

    void elementWasRemoved() const;

    void willChangeRenderer() const;
    void cancelStyleOriginatedAnimations() const;
    void cancelStyleOriginatedAnimations(const WeakStyleOriginatedAnimations&) const;

    void animationWasAdded(WebAnimation&) const;
    void animationWasRemoved(WebAnimation&) const;

    void removeStyleOriginatedAnimationFromListsForOwningElement(WebAnimation&) const;

    void updateCSSAnimations(const RenderStyle* currentStyle, const RenderStyle& afterChangeStyle, const Style::ResolutionContext&, WeakStyleOriginatedAnimations&, Style::IsInDisplayNoneTree) const;
    void updateCSSTransitions(const RenderStyle& currentStyle, const RenderStyle& newStyle, WeakStyleOriginatedAnimations&) const;
    void updateCSSScrollTimelines(const RenderStyle* currentStyle, const RenderStyle& afterChangeStyle) const;
    void updateCSSViewTimelines(const RenderStyle* currentStyle, const RenderStyle& afterChangeStyle) const;
};

class WeakStyleable {
public:
    WeakStyleable() = default;

    explicit operator bool() const { return !!m_element; }

    bool operator==(const WeakStyleable& other) const = default;

    WeakStyleable& operator=(const Styleable& styleable)
    {
        m_element = styleable.element;
        m_pseudoElementIdentifier = styleable.pseudoElementIdentifier;
        return *this;
    }

    WeakStyleable(const Styleable& styleable)
    {
        m_element = styleable.element;
        m_pseudoElementIdentifier = styleable.pseudoElementIdentifier;
    }

    std::optional<Styleable> styleable() const
    {
        if (!m_element)
            return std::nullopt;
        return Styleable(*m_element, m_pseudoElementIdentifier);
    }

    WeakPtr<Element, WeakPtrImplWithEventTargetData> element() const { return m_element; }
    std::optional<Style::PseudoElementIdentifier> pseudoElementIdentifier() const { return m_pseudoElementIdentifier; }

private:
    WeakPtr<Element, WeakPtrImplWithEventTargetData> m_element;
    std::optional<Style::PseudoElementIdentifier> m_pseudoElementIdentifier;
};

} // namespace WebCore
