/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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

namespace WebCore {
    
template<typename PropertyType>
class SVGAnimatedValueProperty : public SVGAnimatedProperty {
public:
    using ValueType = typename PropertyType::ValueType;

    template<typename... Arguments>
    static Ref<SVGAnimatedValueProperty> create(SVGElement* contextElement, Arguments&&... arguments)
    {
        return adoptRef(*new SVGAnimatedValueProperty(contextElement, std::forward<Arguments>(arguments)...));
    }

    ~SVGAnimatedValueProperty()
    {
        m_baseVal->detach();
        if (m_animVal)
            m_animVal->detach();
    }

    // Used by SVGElement::parseAttribute().
    void setBaseValInternal(const ValueType& baseVal)
    {
        m_baseVal->setValue(baseVal);
        if (m_animVal)
            m_animVal->setValue(baseVal);
    }

    // Used by the DOM.
    const Ref<PropertyType>& baseVal() const { return m_baseVal; }

    Ref<PropertyType>& baseVal() { return m_baseVal; }

    // Used by SVGAnimator::progress.
    void setAnimVal(const ValueType& animVal)
    {
        ASSERT(isAnimating() && m_animVal);
        m_animVal->setValue(animVal);
    }

    // Used by the DOM.
    const RefPtr<PropertyType>& animVal() const { return const_cast<SVGAnimatedValueProperty*>(this)->ensureAnimVal(); }

    // Called by SVGAnimatedPropertyAnimator to pass the animVal to the SVGAnimationFunction::progress.
    RefPtr<PropertyType>& animVal() { return ensureAnimVal(); }

    // Used when committing a change from the SVGAnimatedProperty to the attribute.
    String baseValAsString() const override { return m_baseVal->valueAsString(); }

    // Used to apply the SVGAnimator change to the target element.
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
    const ValueType& currentValue() const
    {
        ASSERT_IMPLIES(isAnimating(), m_animVal);
        return (isAnimating() ? *m_animVal : m_baseVal.get()).value();
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
        if (m_animVal)
            m_animVal->setValue(m_baseVal->value());
    }

    // Controlling the instance animation.
    void instanceStartAnimation(SVGAttributeAnimator& animator, SVGAnimatedProperty& animated) override
    {
        if (!isAnimating())
            m_animVal = static_cast<SVGAnimatedValueProperty&>(animated).animVal();
        SVGAnimatedProperty::instanceStartAnimation(animator, animated);
    }

    void instanceStopAnimation(SVGAttributeAnimator& animator) override
    {
        SVGAnimatedProperty::instanceStopAnimation(animator);
        if (!isAnimating())
            m_animVal = nullptr;
    }

protected:
    // The packed arguments are used in PropertyType creation, for example passing
    // SVGLengthMode to SVGLength.
    template<typename... Arguments>
    SVGAnimatedValueProperty(SVGElement* contextElement, Arguments&&... arguments)
        : SVGAnimatedProperty(contextElement)
        , m_baseVal(PropertyType::create(this, SVGPropertyAccess::ReadWrite, ValueType(std::forward<Arguments>(arguments)...)))
    {
    }

    template<typename... Arguments>
    SVGAnimatedValueProperty(SVGElement* contextElement, SVGPropertyAccess access, Arguments&&... arguments)
        : SVGAnimatedProperty(contextElement)
        , m_baseVal(PropertyType::create(this, access, ValueType(std::forward<Arguments>(arguments)...)))
    {
    }

    RefPtr<PropertyType>& ensureAnimVal()
    {
        if (!m_animVal)
            m_animVal = PropertyType::create(this, SVGPropertyAccess::ReadOnly, m_baseVal->value());
        return m_animVal;
    }

    // Called when m_baseVal changes.
    void commitPropertyChange(SVGProperty* property) override
    {
        if (m_animVal)
            m_animVal->setValue(m_baseVal->value());
        SVGAnimatedProperty::commitPropertyChange(property);
    }

    Ref<PropertyType> m_baseVal;
    mutable RefPtr<PropertyType> m_animVal;
};

}
