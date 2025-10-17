/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class SVGElement;

template<typename AnimatedPropertyAnimator1, typename AnimatedPropertyAnimator2>
class SVGAnimatedPropertyPairAnimator : public SVGAttributeAnimator {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(SVGAnimatedPropertyPairAnimator);
public:
    using AnimatedProperty1 = typename AnimatedPropertyAnimator1::AnimatorAnimatedProperty;
    using AnimatedProperty2 = typename AnimatedPropertyAnimator2::AnimatorAnimatedProperty;

    SVGAnimatedPropertyPairAnimator(const QualifiedName& attributeName, Ref<AnimatedProperty1>& animated1, Ref<AnimatedProperty2>& animated2, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
        : SVGAttributeAnimator(attributeName)
        , m_animatedPropertyAnimator1(AnimatedPropertyAnimator1::create(attributeName, animated1, animationMode, calcMode, isAccumulated, isAdditive))
        , m_animatedPropertyAnimator2(AnimatedPropertyAnimator2::create(attributeName, animated2, animationMode, calcMode, isAccumulated, isAdditive))
    {
    }

    void appendAnimatedInstance(Ref<AnimatedProperty1>& animated1, Ref<AnimatedProperty2>& animated2)
    {
        m_animatedPropertyAnimator1->appendAnimatedInstance(animated1);
        m_animatedPropertyAnimator2->appendAnimatedInstance(animated2);
    }

protected:
    void start(SVGElement& targetElement) override
    {
        m_animatedPropertyAnimator1->start(targetElement);
        m_animatedPropertyAnimator2->start(targetElement);
    }

    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) override
    {
        m_animatedPropertyAnimator1->animate(targetElement, progress, repeatCount);
        m_animatedPropertyAnimator2->animate(targetElement, progress, repeatCount);
    }

    void apply(SVGElement& targetElement) override
    {
        applyAnimatedPropertyChange(targetElement);
    }

    void stop(SVGElement& targetElement) override
    {
        m_animatedPropertyAnimator1->stop(targetElement);
        m_animatedPropertyAnimator2->stop(targetElement);
    }

    Ref<AnimatedPropertyAnimator1> m_animatedPropertyAnimator1;
    Ref<AnimatedPropertyAnimator2> m_animatedPropertyAnimator2;
};

#define TZONE_TEMPLATE_PARAMS template<typename AnimatedPropertyAnimator1, typename AnimatedPropertyAnimator2>
#define TZONE_TYPE SVGAnimatedPropertyPairAnimator<AnimatedPropertyAnimator1, AnimatedPropertyAnimator2>

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL_WITH_MULTIPLE_OR_SPECIALIZED_PARAMETERS();

#undef TZONE_TEMPLATE_PARAMS
#undef TZONE_TYPE

} // namespace WebCore
