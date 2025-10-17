/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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

#include "SVGAnimatedProperty.h"
#include "SVGSharedPrimitiveProperty.h"

namespace WebCore {

template<typename PropertyType>
class SVGAnimatedPrimitiveProperty : public SVGAnimatedProperty {
public:
    using ValueType = PropertyType;

    static Ref<SVGAnimatedPrimitiveProperty> create(SVGElement* contextElement)
    {
        return adoptRef(*new SVGAnimatedPrimitiveProperty(contextElement));
    }

    static Ref<SVGAnimatedPrimitiveProperty> create(SVGElement* contextElement, const PropertyType& value)
    {
        return adoptRef(*new SVGAnimatedPrimitiveProperty(contextElement, value));
    }

    // Used by the DOM.
    ExceptionOr<void> setBaseVal(const PropertyType& baseVal)
    {
        m_baseVal->setValue(baseVal);
        commitPropertyChange(nullptr);
        return { };
    }

    // Used by SVGElement::parseAttribute().
    void setBaseValInternal(const PropertyType& baseVal) { m_baseVal->setValue(baseVal); }
    const PropertyType& baseVal() const { return m_baseVal->value(); }

    // Used by SVGAttributeAnimator::progress.
    void setAnimVal(const PropertyType& animVal)
    {
        ASSERT(isAnimating() && m_animVal);
        m_animVal->setValue(animVal);
    }

    const PropertyType& animVal() const
    {
        ASSERT_IMPLIES(isAnimating(), m_animVal);
        return isAnimating() ? m_animVal->value() : m_baseVal->value();
    }

    PropertyType& animVal()
    {
        ASSERT_IMPLIES(isAnimating(), m_animVal);
        return isAnimating() ? m_animVal->value() : m_baseVal->value();
    }

    // Used when committing a change from the SVGAnimatedProperty to the attribute.
    String baseValAsString() const override { return m_baseVal->valueAsString(); }

    // Used to apply the SVGAttributeAnimator change to the target element.
    String animValAsString() const override
    {
        ASSERT(isAnimating() && m_animVal);
        return m_animVal->valueAsString();
    }

    // Managing the relationship with the owner.
    void setDirty() override { m_baseVal->setDirty(); }
    bool isDirty() const override { return m_baseVal->isDirty(); }
    std::optional<String> synchronize() override { return m_baseVal->synchronize(); }

    // Used by RenderSVGElements and DumpRenderTree.
    const PropertyType& currentValue() const
    {
        ASSERT_IMPLIES(isAnimating(), m_animVal);
        return isAnimating() ? m_animVal->value() : m_baseVal->value();
    }

    // Controlling the animation.
    void startAnimation(SVGAttributeAnimator& animator) override
    {
        if (m_animVal)
            m_animVal->setValue(m_baseVal->value());
        else
            ensureAnimVal();
        SVGAnimatedProperty::startAnimation(animator);
    }

    void stopAnimation(SVGAttributeAnimator& animator) override
    {
        SVGAnimatedProperty::stopAnimation(animator);
        if (!isAnimating())
            m_animVal = nullptr;
        else if (m_animVal)
            m_animVal->setValue(m_baseVal->value());
    }

    // Controlling the instance animation.
    void instanceStartAnimation(SVGAttributeAnimator& animator, SVGAnimatedProperty& animated) override
    {
        if (!isAnimating())
            m_animVal = static_cast<SVGAnimatedPrimitiveProperty&>(animated).m_animVal;
        SVGAnimatedProperty::instanceStartAnimation(animator, animated);
    }

    void instanceStopAnimation(SVGAttributeAnimator& animator) override
    {
        SVGAnimatedProperty::instanceStopAnimation(animator);
        if (!isAnimating())
            m_animVal = nullptr;
    }

protected:
    SVGAnimatedPrimitiveProperty(SVGElement* contextElement)
        : SVGAnimatedProperty(contextElement)
        , m_baseVal(SVGSharedPrimitiveProperty<PropertyType>::create())
    {
    }

    SVGAnimatedPrimitiveProperty(SVGElement* contextElement, const PropertyType& value)
        : SVGAnimatedProperty(contextElement)
        , m_baseVal(SVGSharedPrimitiveProperty<PropertyType>::create(value))
    {
    }

    RefPtr<SVGSharedPrimitiveProperty<PropertyType>>& ensureAnimVal()
    {
        if (!m_animVal)
            m_animVal = SVGSharedPrimitiveProperty<PropertyType>::create(m_baseVal->value());
        return m_animVal;
    }

    Ref<SVGSharedPrimitiveProperty<PropertyType>> m_baseVal;
    mutable RefPtr<SVGSharedPrimitiveProperty<PropertyType>> m_animVal;
};

}
