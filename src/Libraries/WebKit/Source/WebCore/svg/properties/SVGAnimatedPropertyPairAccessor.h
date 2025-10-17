/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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

#include "SVGMemberAccessor.h"

namespace WebCore {

template<typename OwnerType, typename AccessorType1, typename AccessorType2>
class SVGAnimatedPropertyPairAccessor : public SVGMemberAccessor<OwnerType> {
    using AnimatedPropertyType1 = typename AccessorType1::AnimatedProperty;
    using AnimatedPropertyType2 = typename AccessorType2::AnimatedProperty;
    using Base = SVGMemberAccessor<OwnerType>;

public:
    SVGAnimatedPropertyPairAccessor(Ref<AnimatedPropertyType1> OwnerType::*property1, Ref<AnimatedPropertyType2> OwnerType::*property2)
        : m_accessor1(property1)
        , m_accessor2(property2)
    {
    }

protected:
    template<typename AccessorType, Ref<AnimatedPropertyType1> OwnerType::*property1, Ref<AnimatedPropertyType2> OwnerType::*property2>
    static SVGMemberAccessor<OwnerType>& singleton()
    {
        static NeverDestroyed<AccessorType> propertyAccessor { property1, property2 };
        return propertyAccessor;
    }

    bool isAnimatedProperty() const override { return true; }

    Ref<AnimatedPropertyType1>& property1(OwnerType& owner) const { return m_accessor1.property(owner); }
    const Ref<AnimatedPropertyType1>& property1(const OwnerType& owner) const { return m_accessor1.property(owner); }

    Ref<AnimatedPropertyType2>& property2(OwnerType& owner) const { return m_accessor2.property(owner); }
    const Ref<AnimatedPropertyType2>& property2(const OwnerType& owner) const { return m_accessor2.property(owner); }

    void detach(const OwnerType& owner) const override
    {
        property1(owner)->detach();
        property2(owner)->detach();
    }

    bool matches(const OwnerType& owner, const SVGAnimatedProperty& animatedProperty) const override
    {
        return m_accessor1.matches(owner, animatedProperty) || m_accessor2.matches(owner, animatedProperty);
    }

    AccessorType1 m_accessor1;
    AccessorType2 m_accessor2;
};

}
