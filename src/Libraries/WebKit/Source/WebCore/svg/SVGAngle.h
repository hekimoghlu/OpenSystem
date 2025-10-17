/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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

#include "SVGAngleValue.h"
#include "SVGValueProperty.h"

namespace WebCore {

class SVGAngle : public SVGValueProperty<SVGAngleValue> {
    using Base = SVGValueProperty<SVGAngleValue>;
    using Base::Base;
    using Base::m_value;

public:
    static Ref<SVGAngle> create(const SVGAngleValue& value = { })
    {
        return adoptRef(*new SVGAngle(value));
    }

    static Ref<SVGAngle> create(SVGPropertyOwner* owner, SVGPropertyAccess access, const SVGAngleValue& value = { })
    {
        return adoptRef(*new SVGAngle(owner, access, value));
    }

    template<typename T>
    static ExceptionOr<Ref<SVGAngle>> create(ExceptionOr<T>&& value)
    {
        if (value.hasException())
            return value.releaseException();
        return adoptRef(*new SVGAngle(value.releaseReturnValue()));
    }

    SVGAngleValue::Type unitType() const
    {
        // Per spec https://svgwg.org/svg2-draft/types.html#__svg__SVGAngle__SVG_ANGLETYPE_UNKNOWN
        if (m_value.unitType() > SVGAngleValue::Type::SVG_ANGLETYPE_GRAD)
            return SVGAngleValue::Type::SVG_ANGLETYPE_UNKNOWN;
        return m_value.unitType();
    }

    ExceptionOr<void> setValueForBindings(float value)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setValue(value);
        commitChange();
        return { };
    }
    
    float valueForBindings() const
    {
        return m_value.value();
    }

    ExceptionOr<void> setValueInSpecifiedUnits(float valueInSpecifiedUnits)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setValueInSpecifiedUnits(valueInSpecifiedUnits);
        commitChange();
        return { };
    }
    
    float valueInSpecifiedUnits() const
    {
        return m_value.valueInSpecifiedUnits();
    }

    ExceptionOr<void> setValueAsString(const String& value)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        auto result = m_value.setValueAsString(value);
        if (result.hasException())
            return result;
        
        commitChange();
        return result;
    }

    String valueAsString() const override
    {
        return m_value.valueAsString();
    }

    ExceptionOr<void> newValueSpecifiedUnits(unsigned short unitType, float valueInSpecifiedUnits)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        auto result = m_value.newValueSpecifiedUnits(unitType, valueInSpecifiedUnits);
        if (result.hasException())
            return result;
        
        commitChange();
        return result;
    }
    
    ExceptionOr<void> convertToSpecifiedUnits(unsigned short unitType)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        auto result = m_value.convertToSpecifiedUnits(unitType);
        if (result.hasException())
            return result;
        
        commitChange();
        return result;
    }
};

} // namespace WebCore
