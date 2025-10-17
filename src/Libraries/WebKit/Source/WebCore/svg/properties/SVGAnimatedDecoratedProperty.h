/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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
#include "SVGDecoratedProperty.h"

namespace WebCore {

template<template <typename, typename> class DecoratedProperty, typename DecorationType>
class SVGAnimatedDecoratedProperty : public SVGAnimatedProperty {
public:
    template<typename PropertyType, typename AnimatedProperty = SVGAnimatedDecoratedProperty>
    static Ref<AnimatedProperty> create(SVGElement* contextElement)
    {
        return adoptRef(*new AnimatedProperty(contextElement, adoptRef(*new DecoratedProperty<DecorationType, PropertyType>())));
    }

    template<typename PropertyType, typename AnimatedProperty = SVGAnimatedDecoratedProperty>
    static Ref<AnimatedProperty> create(SVGElement* contextElement, const PropertyType& value)
    {
        return adoptRef(*new AnimatedProperty(contextElement, DecoratedProperty<DecorationType, PropertyType>::create(value)));
    }

    SVGAnimatedDecoratedProperty(SVGElement* contextElement, Ref<SVGDecoratedProperty<DecorationType>>&& baseVal)
        : SVGAnimatedProperty(contextElement)
        , m_baseVal(WTFMove(baseVal))
    {
    }

    // Used by the DOM.
    ExceptionOr<void> setBaseVal(const DecorationType& baseVal)
    {
        if (!m_baseVal->setValue(baseVal))
            return Exception { ExceptionCode::TypeError };
        commitPropertyChange(nullptr);
        return { };
    }

    // Used by SVGElement::parseAttribute().
    template<typename PropertyType>
    void setBaseValInternal(const PropertyType& baseVal)
    {
        m_baseVal->setValueInternal(static_cast<DecorationType>(baseVal));
        if (m_animVal)
            m_animVal->setValueInternal(static_cast<DecorationType>(baseVal));
    }

    DecorationType baseVal() const { return m_baseVal->value(); }

    // Used by SVGAnimator::progress.
    template<typename PropertyType>
    void setAnimVal(const PropertyType& animVal)
    {
        ASSERT(isAnimating() && m_animVal);
        m_animVal->setValueInternal(static_cast<DecorationType>(animVal));
    }

    template<typename PropertyType = DecorationType>
    PropertyType animVal() const
    {
        ASSERT_IMPLIES(isAnimating(), m_animVal);
        return static_cast<PropertyType>((isAnimating() ? *m_animVal : m_baseVal.get()).value());
    }

    // Used when committing a change from the SVGAnimatedProperty to the attribute.
    String baseValAsString() const override { return m_baseVal->valueAsString(); }

    // Used to apply the SVGAnimator change to the target element.
    String animValAsString() const override
    {
        ASSERT(isAnimating() && !!m_animVal);
        return m_animVal->valueAsString();
    }

    // Managing the relationship with the owner.
    void setDirty() override { m_state = SVGPropertyState::Dirty; }
    bool isDirty() const override { return m_state == SVGPropertyState::Dirty; }
    std::optional<String> synchronize() override
    {
        if (m_state == SVGPropertyState::Clean)
            return std::nullopt;
        m_state = SVGPropertyState::Clean;
        return baseValAsString();
    }

    // Used by RenderSVGElements and DumpRenderTree.
    template<typename PropertyType>
    PropertyType currentValue() const
    {
        ASSERT_IMPLIES(isAnimating(), m_animVal);
        return static_cast<PropertyType>((isAnimating() ? *m_animVal : m_baseVal.get()).valueInternal());
    }

    // Controlling the animation.
    void startAnimation(SVGAttributeAnimator& animator) override
    {
        if (m_animVal)
            m_animVal->setValue(m_baseVal->value());
        else
            m_animVal = m_baseVal->clone();
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
            m_animVal = static_cast<decltype(*this)>(animated).m_animVal;
        SVGAnimatedProperty::instanceStartAnimation(animator, animated);
    }

    void instanceStopAnimation(SVGAttributeAnimator& animator) override
    {
        SVGAnimatedProperty::instanceStopAnimation(animator);
        if (!isAnimating())
            m_animVal = nullptr;
    }

protected:
    Ref<SVGDecoratedProperty<DecorationType>> m_baseVal;
    RefPtr<SVGDecoratedProperty<DecorationType>> m_animVal;
    SVGPropertyState m_state { SVGPropertyState::Clean };
};

}
