/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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

template<typename ListType>
class SVGAnimatedPropertyList : public SVGAnimatedProperty {
public:
    template<typename... Arguments>
    static Ref<SVGAnimatedPropertyList> create(SVGElement* contextElement, Arguments&&... arguments)
    {
        return adoptRef(*new SVGAnimatedPropertyList(contextElement, std::forward<Arguments>(arguments)...));
    }

    ~SVGAnimatedPropertyList()
    {
        m_baseVal->detach();
        if (m_animVal)
            m_animVal->detach();
    }

    // Used by the DOM.
    const Ref<ListType>& baseVal() const { return m_baseVal; }

    // Used by SVGElement::parseAttribute().
    Ref<ListType>& baseVal() { return m_baseVal; }

    // Used by the DOM.
    const RefPtr<ListType>& animVal() const { return const_cast<SVGAnimatedPropertyList*>(this)->ensureAnimVal(); }

    // Called by SVGAnimatedPropertyAnimator to pass the animVal to the SVGAnimationFunction::progress.
    RefPtr<ListType>& animVal() { return ensureAnimVal(); }

    // Used when committing a change from the SVGAnimatedProperty to the attribute.
    String baseValAsString() const override { return m_baseVal->valueAsString(); }

    // Used to apply the SVGAnimator change to the target element.
    String animValAsString() const override
    {
        ASSERT(isAnimating());
        return m_animVal->valueAsString();
    }

    // Managing the relationship with the owner.
    void setDirty() override { m_baseVal->setDirty(); }
    bool isDirty() const override { return m_baseVal->isDirty(); }
    std::optional<String> synchronize() override { return m_baseVal->synchronize(); }

    // Used by RenderSVGElements and DumpRenderTree.
    const ListType& currentValue() const
    {
        ASSERT_IMPLIES(isAnimating(), m_animVal);
        return isAnimating() ? *m_animVal : m_baseVal.get();
    }

    // Controlling the animation.
    void startAnimation(SVGAttributeAnimator& animator) override
    {
        if (m_animVal)
            *m_animVal = m_baseVal;
        else
            ensureAnimVal();
        SVGAnimatedProperty::startAnimation(animator);
    }

    void stopAnimation(SVGAttributeAnimator& animator) override
    {
        SVGAnimatedProperty::stopAnimation(animator);
        if (m_animVal)
            *m_animVal = m_baseVal;
    }

    // Controlling the instance animation.
    void instanceStartAnimation(SVGAttributeAnimator& animator, SVGAnimatedProperty& animated) override
    {
        if (!isAnimating())
            m_animVal = static_cast<SVGAnimatedPropertyList&>(animated).animVal();
        SVGAnimatedProperty::instanceStartAnimation(animator, animated);
    }

    void instanceStopAnimation(SVGAttributeAnimator& animator) override
    {
        SVGAnimatedProperty::instanceStopAnimation(animator);
        if (!isAnimating())
            m_animVal = nullptr;
    }

protected:
    template<typename... Arguments>
    SVGAnimatedPropertyList(SVGElement* contextElement, Arguments&&... arguments)
        : SVGAnimatedProperty(contextElement)
        , m_baseVal(ListType::create(this, SVGPropertyAccess::ReadWrite, std::forward<Arguments>(arguments)...))
    {
    }

    RefPtr<ListType>& ensureAnimVal()
    {
        if (!m_animVal)
            m_animVal = ListType::create(m_baseVal, SVGPropertyAccess::ReadOnly);
        return m_animVal;
    }

    // Called when m_baseVal changes or an item in m_baseVal changes.
    void commitPropertyChange(SVGProperty* property) override
    {
        if (m_animVal)
            *m_animVal = m_baseVal;
        SVGAnimatedProperty::commitPropertyChange(property);
    }

    Ref<ListType> m_baseVal;
    mutable RefPtr<ListType> m_animVal;
};

}
