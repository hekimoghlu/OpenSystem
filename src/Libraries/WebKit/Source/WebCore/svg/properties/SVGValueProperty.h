/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#include "SVGProperty.h"

namespace WebCore {
    
template<typename PropertyType>
class SVGValueProperty : public SVGProperty {
public:
    using ValueType = PropertyType;

    static Ref<SVGValueProperty> create()
    {
        return adoptRef(*new SVGValueProperty());
    }

    // Getter/Setter for the value.
    const PropertyType& value() const { return m_value; }
    void setValue(const PropertyType& value) { m_value = value; }

    // Used by the SVGAnimatedPropertyAnimator to pass m_value to SVGAnimationFunction.
    PropertyType& value() { return m_value; }

protected:
    // Create an initialized property, e.g creating an item to be appended in an SVGList.
    SVGValueProperty(const PropertyType& value)
        : m_value(value)
    {
    }

    // Needed when value should not be copied, e.g. SVGTransformValue.
    SVGValueProperty(PropertyType&& value)
        : m_value(WTFMove(value))
    {
    }

    // Base and default constructor. Do not use "using SVGProperty::SVGProperty" because of Windows and GTK ports.
    SVGValueProperty(SVGPropertyOwner* owner = nullptr, SVGPropertyAccess access = SVGPropertyAccess::ReadWrite)
        : SVGProperty(owner, access)
    {
    }

    // Create an initialized and attached property.
    SVGValueProperty(SVGPropertyOwner* owner, SVGPropertyAccess access, const PropertyType& value)
        : SVGProperty(owner, access)
        , m_value(value)
    {
    }

    PropertyType m_value;
};

}
