/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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

#include "SVGDecoratedProperty.h"

namespace WebCore {

template<typename DecorationType, typename PropertyType>
class SVGDecoratedPrimitive : public SVGDecoratedProperty<DecorationType> {
    using Base = SVGDecoratedProperty<DecorationType>;

public:
    SVGDecoratedPrimitive(const PropertyType& value)
        : m_value(value)
    {
    }

protected:
    using Base::Base;

    void setValueInternal(const DecorationType& value) override { m_value = static_cast<PropertyType>(value); }
    DecorationType valueInternal() const override { return static_cast<DecorationType>(m_value); }

    String valueAsString() const override { return SVGPropertyTraits<PropertyType>::toString(m_value); }

    PropertyType m_value;
};

}
