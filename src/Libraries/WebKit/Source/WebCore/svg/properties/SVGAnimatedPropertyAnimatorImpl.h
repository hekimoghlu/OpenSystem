/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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

#include "HTMLNames.h"
#include "SVGAnimatedPropertyAnimator.h"
#include "SVGAnimatedPropertyImpl.h"
#include "SVGAnimationAdditiveListFunctionImpl.h"
#include "SVGAnimationAdditiveValueFunctionImpl.h"
#include "SVGAnimationDiscreteFunctionImpl.h"

namespace WebCore {

class SVGAnimatedAngleAnimator;
class SVGAnimatedIntegerPairAnimator;
class SVGAnimatedOrientTypeAnimator;

template<typename AnimatedPropertyAnimator1, typename AnimatedPropertyAnimator2>
class SVGAnimatedPropertyPairAnimator;

class SVGAnimatedAngleAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedAngle, SVGAnimationAngleFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedAngleAnimator);
    friend class SVGAnimatedPropertyPairAnimator<SVGAnimatedAngleAnimator, SVGAnimatedOrientTypeAnimator>;
    friend class SVGAnimatedAngleOrientAnimator;
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedAngle, SVGAnimationAngleFunction>;
    using Base::Base;

public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedAngle>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedAngleAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        m_function.animate(targetElement, progress, repeatCount, m_animated->animVal()->value());
    }
};

class SVGAnimatedBooleanAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedBoolean, SVGAnimationBooleanFunction>  {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedBooleanAnimator);
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedBoolean, SVGAnimationBooleanFunction>;

public:
    using Base::Base;

    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedBoolean>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedBooleanAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        bool& animated = m_animated->animVal();
        m_function.animate(targetElement, progress, repeatCount, animated);
    }
};

template<typename EnumType>
class SVGAnimatedEnumerationAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedEnumeration, SVGAnimationEnumerationFunction<EnumType>> {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(SVGAnimatedEnumerationAnimator);
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedEnumeration, SVGAnimationEnumerationFunction<EnumType>>;
    using Base::Base;
    using Base::m_animated;
    using Base::m_function;

public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedEnumeration>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedEnumerationAnimator<EnumType>(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        EnumType animated;
        m_function.animate(targetElement, progress, repeatCount, animated);
        m_animated->template setAnimVal<EnumType>(animated);
    }
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename EnumType>, SVGAnimatedEnumerationAnimator<EnumType>);

class SVGAnimatedIntegerAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedInteger, SVGAnimationIntegerFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedIntegerAnimator);
    friend class SVGAnimatedPropertyPairAnimator<SVGAnimatedIntegerAnimator, SVGAnimatedIntegerAnimator>;
    friend class SVGAnimatedIntegerPairAnimator;
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedInteger, SVGAnimationIntegerFunction>;

public:
    using Base::Base;

    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedInteger>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedIntegerAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        m_function.animate(targetElement, progress, repeatCount, m_animated->animVal());
    }
};

class SVGAnimatedLengthAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedLength, SVGAnimationLengthFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedLengthAnimator);
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedLength, SVGAnimationLengthFunction>;

public:
    SVGAnimatedLengthAnimator(const QualifiedName& attributeName, Ref<SVGAnimatedLength>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive, SVGLengthMode lengthMode)
        : Base(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive, lengthMode)
    {
    }

    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedLength>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive, SVGLengthMode lengthMode)
    {
        return adoptRef(*new SVGAnimatedLengthAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive, lengthMode));
    }

private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        m_function.animate(targetElement, progress, repeatCount, m_animated->animVal()->value());
    }
};

class SVGAnimatedLengthListAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedLengthList, SVGAnimationLengthListFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedLengthListAnimator);
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedLengthList, SVGAnimationLengthListFunction>;

public:
    SVGAnimatedLengthListAnimator(const QualifiedName& attributeName, Ref<SVGAnimatedLengthList>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive, SVGLengthMode lengthMode)
        : Base(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive, lengthMode)
    {
    }

    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedLengthList>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive, SVGLengthMode lengthMode)
    {
        return adoptRef(*new SVGAnimatedLengthListAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive, lengthMode));
    }

private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        m_function.animate(targetElement, progress, repeatCount, m_animated->animVal());
    }
};

class SVGAnimatedNumberAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedNumber, SVGAnimationNumberFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedNumberAnimator);
    friend class SVGAnimatedPropertyPairAnimator<SVGAnimatedNumberAnimator, SVGAnimatedNumberAnimator>;
    friend class SVGAnimatedNumberPairAnimator;
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedNumber, SVGAnimationNumberFunction>;
    using Base::Base;

public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedNumber>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedNumberAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        m_function.animate(targetElement, progress, repeatCount, m_animated->animVal());
    }
};

class SVGAnimatedNumberListAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedNumberList, SVGAnimationNumberListFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedNumberListAnimator);
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedNumberList, SVGAnimationNumberListFunction>;
    using Base::Base;
    
public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedNumberList>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedNumberListAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }
    
private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        m_function.animate(targetElement, progress, repeatCount, m_animated->animVal());
    }
};

class SVGAnimatedPathSegListAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedPathSegList, SVGAnimationPathSegListFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedPathSegListAnimator);
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedPathSegList, SVGAnimationPathSegListFunction>;
    using Base::Base;

public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedPathSegList>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedPathSegListAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        m_animated->animVal()->pathByteStreamWillChange();
        m_function.animate(targetElement, progress, repeatCount, m_animated->animVal()->pathByteStream());
    }
};

class SVGAnimatedPointListAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedPointList, SVGAnimationPointListFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedPointListAnimator);
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedPointList, SVGAnimationPointListFunction>;
    using Base::Base;
    
public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedPointList>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedPointListAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }
    
private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        m_function.animate(targetElement, progress, repeatCount, m_animated->animVal());
    }
};

class SVGAnimatedOrientTypeAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedOrientType, SVGAnimationOrientTypeFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedOrientTypeAnimator);
    friend class SVGAnimatedPropertyPairAnimator<SVGAnimatedAngleAnimator, SVGAnimatedOrientTypeAnimator>;
    friend class SVGAnimatedAngleOrientAnimator;
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedOrientType, SVGAnimationOrientTypeFunction>;
    using Base::Base;

public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedOrientType>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedOrientTypeAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        SVGMarkerOrientType animated;
        m_function.animate(targetElement, progress, repeatCount, animated);
        m_animated->setAnimVal(animated);
    }
};

class SVGAnimatedPreserveAspectRatioAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedPreserveAspectRatio, SVGAnimationPreserveAspectRatioFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedPreserveAspectRatioAnimator);
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedPreserveAspectRatio, SVGAnimationPreserveAspectRatioFunction>;
    using Base::Base;

public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedPreserveAspectRatio>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedPreserveAspectRatioAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        SVGPreserveAspectRatioValue& animated = m_animated->animVal()->value();
        m_function.animate(targetElement, progress, repeatCount, animated);
    }
};

class SVGAnimatedRectAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedRect, SVGAnimationRectFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedRectAnimator);
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedRect, SVGAnimationRectFunction>;

public:
    using Base::Base;

    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedRect>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedRectAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        m_function.animate(targetElement, progress, repeatCount, m_animated->animVal()->value());
    }
};

class SVGAnimatedStringAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedString, SVGAnimationStringFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedStringAnimator);
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedString, SVGAnimationStringFunction>;
    using Base::Base;

public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedString>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedStringAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    bool isAnimatedStyleClassAnimator() const
    {
        return m_attributeName.matches(HTMLNames::classAttr);
    }

    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        String& animated = m_animated->animVal();
        m_function.animate(targetElement, progress, repeatCount, animated);
    }
    
    void apply(SVGElement& targetElement) final
    {
        Base::apply(targetElement);
        if (isAnimatedStyleClassAnimator())
            invalidateStyle(targetElement);
    }
    
    void stop(SVGElement& targetElement) final
    {
        if (!m_animated->isAnimating())
            return;

        Base::stop(targetElement);
        if (isAnimatedStyleClassAnimator())
            invalidateStyle(targetElement);
    }
};

class SVGAnimatedTransformListAnimator final : public SVGAnimatedPropertyAnimator<SVGAnimatedTransformList, SVGAnimationTransformListFunction> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAnimatedTransformListAnimator);
    using Base = SVGAnimatedPropertyAnimator<SVGAnimatedTransformList, SVGAnimationTransformListFunction>;
    using Base::Base;

public:
    static auto create(const QualifiedName& attributeName, Ref<SVGAnimatedTransformList>& animated, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGAnimatedTransformListAnimator(attributeName, animated, animationMode, calcMode, isAccumulated, isAdditive));
    }

private:
    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) final
    {
        m_function.animate(targetElement, progress, repeatCount, m_animated->animVal());
    }
};

} // namespace WebCore
