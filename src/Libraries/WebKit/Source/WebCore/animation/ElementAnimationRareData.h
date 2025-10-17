/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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

#include "KeyframeEffectStack.h"
#include "PseudoElementIdentifier.h"
#include "WebAnimationTypes.h"

namespace WebCore {

class CSSAnimation;
class CSSTransition;
class RenderStyle;
class WebAnimation;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ElementAnimationRareData);
class ElementAnimationRareData {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(ElementAnimationRareData);
    WTF_MAKE_NONCOPYABLE(ElementAnimationRareData);
public:
    explicit ElementAnimationRareData();
    ~ElementAnimationRareData();

    KeyframeEffectStack* keyframeEffectStack() { return m_keyframeEffectStack.get(); }
    KeyframeEffectStack& ensureKeyframeEffectStack();

    AnimationCollection& animations() { return m_animations; }
    CSSAnimationCollection& animationsCreatedByMarkup() { return m_animationsCreatedByMarkup; }
    void setAnimationsCreatedByMarkup(CSSAnimationCollection&&);
    AnimatableCSSPropertyToTransitionMap& completedTransitionsByProperty() { return m_completedTransitionsByProperty; }
    AnimatableCSSPropertyToTransitionMap& runningTransitionsByProperty() { return m_runningTransitionsByProperty; }
    const RenderStyle* lastStyleChangeEventStyle() const { return m_lastStyleChangeEventStyle.get(); }
    void setLastStyleChangeEventStyle(std::unique_ptr<const RenderStyle>&&);
    void cssAnimationsDidUpdate() { m_hasPendingKeyframesUpdate = false; }
    void keyframesRuleDidChange() { m_hasPendingKeyframesUpdate = true; }
    bool hasPendingKeyframesUpdate() const { return m_hasPendingKeyframesUpdate; }
    bool hasPropertiesOverridenAfterAnimation() const { return m_hasPropertiesOverridenAfterAnimation; }
    void setHasPropertiesOverridenAfterAnimation(bool value) { m_hasPropertiesOverridenAfterAnimation = value; }

private:
    std::unique_ptr<KeyframeEffectStack> m_keyframeEffectStack;
    std::unique_ptr<const RenderStyle> m_lastStyleChangeEventStyle;
    AnimationCollection m_animations;
    CSSAnimationCollection m_animationsCreatedByMarkup;
    AnimatableCSSPropertyToTransitionMap m_completedTransitionsByProperty;
    AnimatableCSSPropertyToTransitionMap m_runningTransitionsByProperty;
    bool m_hasPendingKeyframesUpdate { false };
    bool m_hasPropertiesOverridenAfterAnimation { false };
};

} // namespace WebCore

