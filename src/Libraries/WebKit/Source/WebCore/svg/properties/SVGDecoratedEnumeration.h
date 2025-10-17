/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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

#include "SVGDecoratedPrimitive.h"

namespace WebCore {

template<typename DecorationType, typename EnumType>
class SVGDecoratedEnumeration : public SVGDecoratedPrimitive<DecorationType, EnumType> {
    using Base = SVGDecoratedPrimitive<DecorationType, EnumType>;
    using Base::Base;
    using Base::m_value;

public:
    static auto create(const EnumType& value)
    {
        static_assert(std::is_integral<DecorationType>::value, "DecorationType form enum should be integral.");
        return adoptRef(*new SVGDecoratedEnumeration(value));
    }

private:
    bool setValue(const DecorationType& value) override
    {
        if (!value || value > SVGIDLEnumLimits<EnumType>::highestExposedEnumValue())
            return false;
        Base::setValueInternal(value);
        return true;
    }

    DecorationType value() const override
    {
        if (Base::value() > SVGIDLEnumLimits<EnumType>::highestExposedEnumValue())
            return m_outOfRangeEnumValue;
        return Base::value();
    }

    Ref<SVGDecoratedProperty<DecorationType>> clone() override
    {
        return create(m_value);
    }

    static const DecorationType m_outOfRangeEnumValue = 0;
};

}
