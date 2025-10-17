/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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

#include "SVGPropertyAnimator.h"
#include "SVGPropertyTraits.h"
#include "SVGValueProperty.h"

namespace WebCore {

template<typename PropertyType, typename AnimationFunction>
class SVGPrimitivePropertyAnimator : public SVGPropertyAnimator<AnimationFunction> {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(SVGPrimitivePropertyAnimator);
    using Base = SVGPropertyAnimator<AnimationFunction>;
    using ValuePropertyType = SVGValueProperty<PropertyType>;
    using Base::Base;
    using Base::applyAnimatedStylePropertyChange;
    using Base::computeCSSPropertyValue;
    using Base::m_attributeName;
    using Base::m_function;
    
public:
    static auto create(const QualifiedName& attributeName, Ref<SVGProperty>&& property, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGPrimitivePropertyAnimator(attributeName, WTFMove(property), animationMode, calcMode, isAccumulated, isAdditive));
    }
    
    template<typename... Arguments>
    SVGPrimitivePropertyAnimator(const QualifiedName& attributeName, Ref<SVGProperty>&& property, Arguments&&... arguments)
        : Base(attributeName, std::forward<Arguments>(arguments)...)
        , m_property(static_reference_cast<ValuePropertyType>(WTFMove(property)))
    {
    }

    void start(SVGElement& targetElement) override
    {
        String baseValue = computeCSSPropertyValue(targetElement, cssPropertyID(m_attributeName.localName()));
        m_property->setValue(SVGPropertyTraits<PropertyType>::fromString(baseValue));
    }

    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) override
    {
        PropertyType& animated = m_property->value();
        m_function.animate(targetElement, progress, repeatCount, animated);
    }

    void apply(SVGElement& targetElement) override
    {
        applyAnimatedStylePropertyChange(targetElement, SVGPropertyTraits<PropertyType>::toString(m_property->value()));
    }

protected:
    Ref<ValuePropertyType> m_property;
};

#define TZONE_TEMPLATE_PARAMS template<typename PropertyType, typename AnimationFunction>
#define TZONE_TYPE SVGPrimitivePropertyAnimator<PropertyType, AnimationFunction>

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL_WITH_MULTIPLE_OR_SPECIALIZED_PARAMETERS();

#undef TZONE_TEMPLATE_PARAMS
#undef TZONE_TYPE

} // namespace WebCore
