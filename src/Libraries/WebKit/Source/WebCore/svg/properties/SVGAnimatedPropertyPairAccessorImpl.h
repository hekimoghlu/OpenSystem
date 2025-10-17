/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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

#include "SVGAnimatedPropertyAccessorImpl.h"
#include "SVGAnimatedPropertyAnimatorImpl.h"
#include "SVGAnimatedPropertyImpl.h"
#include "SVGAnimatedPropertyPairAccessor.h"
#include "SVGAnimatedPropertyPairAnimatorImpl.h"
#include "SVGNames.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

template<typename OwnerType>
class SVGAnimatedAngleOrientAccessor final : public SVGAnimatedPropertyPairAccessor<OwnerType, SVGAnimatedAngleAccessor<OwnerType>, SVGAnimatedOrientTypeAccessor<OwnerType>> {
    using Base = SVGAnimatedPropertyPairAccessor<OwnerType, SVGAnimatedAngleAccessor<OwnerType>, SVGAnimatedOrientTypeAccessor<OwnerType>>;
    using Base::property1;
    using Base::property2;
    using Base::m_accessor1;
    using Base::m_accessor2;

public:
    using Base::Base;
    template<Ref<SVGAnimatedAngle> OwnerType::*property1, Ref<SVGAnimatedOrientType> OwnerType::*property2>
    constexpr static const SVGMemberAccessor<OwnerType>& singleton() { return Base::template singleton<SVGAnimatedAngleOrientAccessor, property1, property2>(); }

private:
    void setDirty(const OwnerType& owner, SVGAnimatedProperty& animatedProperty) const final
    {
        auto type = property2(owner)->baseVal();
        if (m_accessor1.matches(owner, animatedProperty) && type != SVGMarkerOrientAngle)
            property2(owner)->setBaseValInternal(SVGMarkerOrientAngle);
        else if (m_accessor2.matches(owner, animatedProperty) && type != SVGMarkerOrientAngle)
            property1(owner)->setBaseValInternal({ });
        animatedProperty.setDirty();
    }

    std::optional<String> synchronize(const OwnerType& owner) const final
    {
        bool dirty1 = property1(owner)->isDirty();
        bool dirty2 = property2(owner)->isDirty();
        if (!(dirty1 || dirty2))
            return std::nullopt;

        auto type = property2(owner)->baseVal();

        String string1 = dirty1 ? *property1(owner)->synchronize() : property1(owner)->baseValAsString();
        String string2 = dirty2 ? *property2(owner)->synchronize() : property2(owner)->baseValAsString();
        return type == SVGMarkerOrientAuto || type == SVGMarkerOrientAutoStartReverse ? string2 : string1;
    }

    RefPtr<SVGAttributeAnimator> createAnimator(OwnerType& owner, const QualifiedName& attributeName, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive) const final
    {
        return SVGAnimatedAngleOrientAnimator::create(attributeName, property1(owner), property2(owner), animationMode, calcMode, isAccumulated, isAdditive);
    }

    void appendAnimatedInstance(OwnerType& owner, SVGAttributeAnimator& animator) const final
    {
        static_cast<SVGAnimatedAngleOrientAnimator&>(animator).appendAnimatedInstance(property1(owner), property2(owner));
    }
};

template<typename OwnerType>
class SVGAnimatedIntegerPairAccessor final : public SVGAnimatedPropertyPairAccessor<OwnerType, SVGAnimatedIntegerAccessor<OwnerType>, SVGAnimatedIntegerAccessor<OwnerType>> {
    using Base = SVGAnimatedPropertyPairAccessor<OwnerType, SVGAnimatedIntegerAccessor<OwnerType>, SVGAnimatedIntegerAccessor<OwnerType>>;
    using Base::property1;
    using Base::property2;

public:
    using Base::Base;
    template<Ref<SVGAnimatedInteger> OwnerType::*property1, Ref<SVGAnimatedInteger> OwnerType::*property2>
    constexpr static const SVGMemberAccessor<OwnerType>& singleton() { return Base::template singleton<SVGAnimatedIntegerPairAccessor, property1, property2>(); }

private:
    std::optional<String> synchronize(const OwnerType& owner) const final
    {
        bool dirty1 = property1(owner)->isDirty();
        bool dirty2 = property2(owner)->isDirty();
        if (!(dirty1 || dirty2))
            return std::nullopt;

        String string1 = dirty1 ? *property1(owner)->synchronize() : property1(owner)->baseValAsString();
        String string2 = dirty2 ? *property2(owner)->synchronize() : property2(owner)->baseValAsString();
        return string1 == string2 ? string1 : makeString(string1, ", "_s, string2);
    }

    RefPtr<SVGAttributeAnimator> createAnimator(OwnerType& owner, const QualifiedName& attributeName, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive) const final
    {
        return SVGAnimatedIntegerPairAnimator::create(attributeName, property1(owner), property2(owner), animationMode, calcMode, isAccumulated, isAdditive);
    }

    void appendAnimatedInstance(OwnerType& owner, SVGAttributeAnimator& animator) const final
    {
        static_cast<SVGAnimatedIntegerPairAnimator&>(animator).appendAnimatedInstance(property1(owner), property2(owner));
    }
};

template<typename OwnerType>
class SVGAnimatedNumberPairAccessor final : public SVGAnimatedPropertyPairAccessor<OwnerType, SVGAnimatedNumberAccessor<OwnerType>, SVGAnimatedNumberAccessor<OwnerType>> {
    using Base = SVGAnimatedPropertyPairAccessor<OwnerType, SVGAnimatedNumberAccessor<OwnerType>, SVGAnimatedNumberAccessor<OwnerType>>;
    using Base::property1;
    using Base::property2;

public:
    using Base::Base;
    template<Ref<SVGAnimatedNumber> OwnerType::*property1, Ref<SVGAnimatedNumber> OwnerType::*property2 >
    constexpr static const SVGMemberAccessor<OwnerType>& singleton() { return Base::template singleton<SVGAnimatedNumberPairAccessor, property1, property2>(); }

private:
    std::optional<String> synchronize(const OwnerType& owner) const final
    {
        bool dirty1 = property1(owner)->isDirty();
        bool dirty2 = property2(owner)->isDirty();
        if (!(dirty1 || dirty2))
            return std::nullopt;

        String string1 = dirty1 ? *property1(owner)->synchronize() : property1(owner)->baseValAsString();
        String string2 = dirty2 ? *property2(owner)->synchronize() : property2(owner)->baseValAsString();
        return string1 == string2 ? string1 : makeString(string1, ", "_s, string2);
    }

    RefPtr<SVGAttributeAnimator> createAnimator(OwnerType& owner, const QualifiedName& attributeName, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive) const final
    {
        return SVGAnimatedNumberPairAnimator::create(attributeName, property1(owner), property2(owner), animationMode, calcMode, isAccumulated, isAdditive);
    }

    void appendAnimatedInstance(OwnerType& owner, SVGAttributeAnimator& animator) const final
    {
        static_cast<SVGAnimatedNumberPairAnimator&>(animator).appendAnimatedInstance(property1(owner), property2(owner));
    }
};

}
