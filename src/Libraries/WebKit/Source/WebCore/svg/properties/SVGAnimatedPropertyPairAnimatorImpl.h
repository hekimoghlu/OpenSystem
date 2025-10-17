/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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

#include "SVGAnimatedPropertyImpl.h"
#include "SVGAnimatedPropertyPairAnimator.h"
#include "SVGMarkerTypes.h"

namespace WebCore {

class SVGElement;

class SVGAnimatedAngleOrientAnimator final : public SVGAnimatedPropertyPairAnimator<SVGAnimatedAngleAnimator, SVGAnimatedOrientTypeAnimator> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedAngleOrientAnimator);
    using Base = SVGAnimatedPropertyPairAnimator<SVGAnimatedAngleAnimator, SVGAnimatedOrientTypeAnimator>;
    using Base::Base;

public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedAngle>& animated1, Ref<SVGAnimatedOrientType>& animated2, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedAngleOrientAnimator(attributeName, animated1, animated2, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void setFromAndToValues(SVGElement&, const String& from, const String& to) final
    {
        auto pairFrom = SVGPropertyTraits<std::pair<SVGAngleValue, SVGMarkerOrientType>>::fromString(from);
        auto pairTo = SVGPropertyTraits<std::pair<SVGAngleValue, SVGMarkerOrientType>>::fromString(to);

        m_animatedPropertyAnimator1->m_function.m_from = pairFrom.first;
        m_animatedPropertyAnimator1->m_function.m_to = pairTo.first;

        m_animatedPropertyAnimator2->m_function.m_from = pairFrom.second;
        m_animatedPropertyAnimator2->m_function.m_to = pairTo.second;
    }

    void setFromAndByValues(SVGElement& targetElement, const String& from, const String& by) final
    {
        setFromAndToValues(targetElement, from, by);
        if (m_animatedPropertyAnimator2->m_function.m_from != SVGMarkerOrientAngle || m_animatedPropertyAnimator2->m_function.m_to != SVGMarkerOrientAngle)
            return;
        m_animatedPropertyAnimator1->m_function.addFromAndToValues(targetElement);
    }

    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        if (m_animatedPropertyAnimator2->m_function.m_from != m_animatedPropertyAnimator2->m_function.m_to) {
            // Discrete animation - no linear interpolation possible between values (e.g. auto to angle).
            m_animatedPropertyAnimator2->animate(targetElement, progress, repeatCount);

            SVGAngleValue animatedAngle;
            if (progress < 0.5f && m_animatedPropertyAnimator2->m_function.m_from == SVGMarkerOrientAngle)
                animatedAngle = m_animatedPropertyAnimator1->m_function.m_from;
            else if (progress >= 0.5f && m_animatedPropertyAnimator2->m_function.m_to == SVGMarkerOrientAngle)
                animatedAngle = m_animatedPropertyAnimator1->m_function.m_to;

            m_animatedPropertyAnimator1->m_animated->setAnimVal(animatedAngle);
            return;
        }

        if (m_animatedPropertyAnimator2->m_function.m_from == SVGMarkerOrientAngle) {
            // Regular from- toangle animation, with support for smooth interpolation, and additive and accumulated animation.
            m_animatedPropertyAnimator2->m_animated->setAnimVal(SVGMarkerOrientAngle);

            m_animatedPropertyAnimator1->animate(targetElement, progress, repeatCount);
            return;
        }

        // auto, auto-start-reverse, or unknown.
        m_animatedPropertyAnimator1->m_animated->animVal()->value().setValue(0);

        if (m_animatedPropertyAnimator2->m_function.m_from == SVGMarkerOrientAuto || m_animatedPropertyAnimator2->m_function.m_from == SVGMarkerOrientAutoStartReverse)
            m_animatedPropertyAnimator2->m_animated->setAnimVal(m_animatedPropertyAnimator2->m_function.m_from);
        else
            m_animatedPropertyAnimator2->m_animated->setAnimVal(SVGMarkerOrientUnknown);
    }

    void stop(SVGElement& targetElement) final
    {
        if (!m_animatedPropertyAnimator1->m_animated->isAnimating())
            return;
        apply(targetElement);
        Base::stop(targetElement);
    }
};

class SVGAnimatedIntegerPairAnimator final : public SVGAnimatedPropertyPairAnimator<SVGAnimatedIntegerAnimator, SVGAnimatedIntegerAnimator> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedIntegerPairAnimator);
    using Base = SVGAnimatedPropertyPairAnimator<SVGAnimatedIntegerAnimator, SVGAnimatedIntegerAnimator>;
    using Base::Base;

public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedInteger>& animated1, Ref<SVGAnimatedInteger>& animated2, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedIntegerPairAnimator(attributeName, animated1, animated2, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void setFromAndToValues(SVGElement&, const String& from, const String& to) final
    {
        auto pairFrom = SVGPropertyTraits<std::pair<int, int>>::fromString(from);
        auto pairTo = SVGPropertyTraits<std::pair<int, int>>::fromString(to);

        m_animatedPropertyAnimator1->m_function.m_from = pairFrom.first;
        m_animatedPropertyAnimator1->m_function.m_to = pairTo.first;

        m_animatedPropertyAnimator2->m_function.m_from = pairFrom.second;
        m_animatedPropertyAnimator2->m_function.m_to = pairTo.second;
    }

    void setFromAndByValues(SVGElement&, const String& from, const String& by) final
    {
        auto pairFrom = SVGPropertyTraits<std::pair<int, int>>::fromString(from);
        auto pairBy = SVGPropertyTraits<std::pair<int, int>>::fromString(by);

        m_animatedPropertyAnimator1->m_function.m_from = pairFrom.first;
        m_animatedPropertyAnimator1->m_function.m_to = pairFrom.first + pairBy.first;

        m_animatedPropertyAnimator2->m_function.m_from = pairFrom.second;
        m_animatedPropertyAnimator2->m_function.m_to = pairFrom.second + pairBy.second;
    }

    void setToAtEndOfDurationValue(const String& toAtEndOfDuration) final
    {
        auto pairToAtEndOfDuration = SVGPropertyTraits<std::pair<int, int>>::fromString(toAtEndOfDuration);
        m_animatedPropertyAnimator1->m_function.m_toAtEndOfDuration = pairToAtEndOfDuration.first;
        m_animatedPropertyAnimator2->m_function.m_toAtEndOfDuration = pairToAtEndOfDuration.second;
    }
};

class SVGAnimatedNumberPairAnimator final : public SVGAnimatedPropertyPairAnimator<SVGAnimatedNumberAnimator, SVGAnimatedNumberAnimator> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedNumberPairAnimator);
    using Base = SVGAnimatedPropertyPairAnimator<SVGAnimatedNumberAnimator, SVGAnimatedNumberAnimator>;
    using Base::Base;

public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedNumber>& animated1, Ref<SVGAnimatedNumber>& animated2, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedNumberPairAnimator(attributeName, animated1, animated2, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void setFromAndToValues(SVGElement&, const String& from, const String& to) final
    {
        auto pairFrom = SVGPropertyTraits<std::pair<float, float>>::fromString(from);
        auto pairTo = SVGPropertyTraits<std::pair<float, float>>::fromString(to);
        
        m_animatedPropertyAnimator1->m_function.m_from = pairFrom.first;
        m_animatedPropertyAnimator1->m_function.m_to = pairTo.first;
        
        m_animatedPropertyAnimator2->m_function.m_from = pairFrom.second;
        m_animatedPropertyAnimator2->m_function.m_to = pairTo.second;
    }

    void setFromAndByValues(SVGElement&, const String& from, const String& by) final
    {
        auto pairFrom = SVGPropertyTraits<std::pair<float, float>>::fromString(from);
        auto pairBy = SVGPropertyTraits<std::pair<float, float>>::fromString(by);
        
        m_animatedPropertyAnimator1->m_function.m_from = pairFrom.first;
        m_animatedPropertyAnimator1->m_function.m_to = pairFrom.first + pairBy.first;
        
        m_animatedPropertyAnimator2->m_function.m_from = pairFrom.second;
        m_animatedPropertyAnimator2->m_function.m_to = pairFrom.second + pairBy.second;
    }

    void setToAtEndOfDurationValue(const String& toAtEndOfDuration) final
    {
        auto pairToAtEndOfDuration = SVGPropertyTraits<std::pair<float, float>>::fromString(toAtEndOfDuration);
        m_animatedPropertyAnimator1->m_function.m_toAtEndOfDuration = pairToAtEndOfDuration.first;
        m_animatedPropertyAnimator2->m_function.m_toAtEndOfDuration = pairToAtEndOfDuration.second;
    }
};

} // namespace WebCore
