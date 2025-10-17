/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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

#include "SVGAttributeAnimator.h"

namespace WebCore {

class SVGElement;

template<typename AnimatedProperty, typename AnimationFunction>
class SVGAnimatedPropertyAnimator : public SVGAttributeAnimator {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(SVGAnimatedPropertyAnimator);
public:
    using AnimatorAnimatedProperty = AnimatedProperty;

    template<typename... Arguments>
    SVGAnimatedPropertyAnimator(const QualifiedName& attributeName, Ref<AnimatedProperty>& animated, Arguments&&... arguments)
        : SVGAttributeAnimator(attributeName)
        , m_animated(animated.copyRef())
        , m_function(std::forward<Arguments>(arguments)...)
    {
    }

    void appendAnimatedInstance(Ref<AnimatedProperty>& animated)
    {
        m_animatedInstances.append(animated.copyRef());
    }

    bool isDiscrete() const override { return m_function.isDiscrete(); }

    void setFromAndToValues(SVGElement& targetElement, const String& from, const String& to) override
    {
        m_function.setFromAndToValues(targetElement, from, to);
    }

    void setFromAndByValues(SVGElement& targetElement, const String& from, const String& by) override
    {
        m_function.setFromAndByValues(targetElement, from, by);
    }

    void setToAtEndOfDurationValue(const String& toAtEndOfDuration) override
    {
        m_function.setToAtEndOfDurationValue(toAtEndOfDuration);
    }

    void start(SVGElement&) override
    {
        m_animated->startAnimation(*this);
        for (auto& instance : m_animatedInstances)
            instance->instanceStartAnimation(*this, m_animated);
    }

    void apply(SVGElement& targetElement) override
    {
        if (isAnimatedStylePropertyAnimator(targetElement))
            applyAnimatedStylePropertyChange(targetElement, m_animated->animValAsString());
        applyAnimatedPropertyChange(targetElement);
    }

    void stop(SVGElement& targetElement) override
    {
        if (!m_animated->isAnimating())
            return;

        m_animated->stopAnimation(*this);
        for (auto& instance : m_animatedInstances)
            instance->instanceStopAnimation(*this);

        applyAnimatedPropertyChange(targetElement);
        if (isAnimatedStylePropertyAnimator(targetElement))
            removeAnimatedStyleProperty(targetElement);
    }

    std::optional<float> calculateDistance(SVGElement& targetElement, const String& from, const String& to) const override
    {
        return m_function.calculateDistance(targetElement, from, to);
    }

protected:
    Ref<AnimatedProperty> m_animated;
    Vector<Ref<AnimatedProperty>> m_animatedInstances;
    AnimationFunction m_function;
};

#define TZONE_TEMPLATE_PARAMS template<typename AnimatedProperty, typename AnimationFunction>
#define TZONE_TYPE SVGAnimatedPropertyAnimator<AnimatedProperty, AnimationFunction>

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL_WITH_MULTIPLE_OR_SPECIALIZED_PARAMETERS();

#undef TZONE_TEMPLATE_PARAMS
#undef TZONE_TYPE

} // namespace WebCore
