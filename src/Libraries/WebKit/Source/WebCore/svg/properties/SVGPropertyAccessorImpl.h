/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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

#include "SVGPropertyAccessor.h"
#include "SVGStringList.h"
#include "SVGTests.h"

namespace WebCore {

template<typename OwnerType>
class SVGConditionalProcessingAttributeAccessor final : public SVGMemberAccessor<OwnerType> {
    using Base = SVGMemberAccessor<OwnerType>;

public:
    SVGConditionalProcessingAttributeAccessor(Ref<SVGStringList> SVGConditionalProcessingAttributes::*property)
        : m_property(property)
    {
    }

    Ref<SVGStringList>& property(OwnerType& owner) const { return owner.conditionalProcessingAttributes().*m_property; }
    const Ref<SVGStringList>& property(const OwnerType& owner) const { return const_cast<OwnerType&>(owner).conditionalProcessingAttributes().*m_property; }

    void detach(const OwnerType& owner) const override
    {
        property(owner)->detach();
    }

    std::optional<String> synchronize(const OwnerType& owner) const override
    {
        return property(owner)->synchronize();
    }

    bool matches(const OwnerType& owner, const SVGProperty& property) const override
    {
        return this->property(owner).ptr() == &property;
    }

    template<Ref<SVGStringList> SVGConditionalProcessingAttributes::*>
    static const SVGMemberAccessor<OwnerType>& singleton();

private:
    Ref<SVGStringList> SVGConditionalProcessingAttributes::*m_property;
};

template<typename OwnerType>
template<Ref<SVGStringList> SVGConditionalProcessingAttributes::*member>
const SVGMemberAccessor<OwnerType>& SVGConditionalProcessingAttributeAccessor<OwnerType>::singleton()
{
    static NeverDestroyed<SVGConditionalProcessingAttributeAccessor<OwnerType>> propertyAccessor { member };
    return propertyAccessor;
}

template<typename OwnerType>
class SVGTransformListAccessor final : public SVGPropertyAccessor<OwnerType, SVGTransformList> {
    using Base = SVGPropertyAccessor<OwnerType, SVGTransformList>;

public:
    using Base::Base;
    template<Ref<SVGTransformList> OwnerType::*property>
    constexpr static const SVGMemberAccessor<OwnerType>& singleton() { return Base::template singleton<SVGTransformListAccessor, property>(); }
};

}
