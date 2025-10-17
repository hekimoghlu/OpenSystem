/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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

#include "SVGLengthContext.h"
#include "SVGValueProperty.h"

namespace WebCore {

class SVGLength : public SVGValueProperty<SVGLengthValue> {
    using Base = SVGValueProperty<SVGLengthValue>;
    using Base::Base;
    using Base::m_value;

public:
    // Forward declare these enums in the w3c naming scheme, for IDL generation
    enum {
        SVG_LENGTHTYPE_UNKNOWN      = static_cast<unsigned>(SVGLengthType::Unknown),
        SVG_LENGTHTYPE_NUMBER       = static_cast<unsigned>(SVGLengthType::Number),
        SVG_LENGTHTYPE_PERCENTAGE   = static_cast<unsigned>(SVGLengthType::Percentage),
        SVG_LENGTHTYPE_EMS          = static_cast<unsigned>(SVGLengthType::Ems),
        SVG_LENGTHTYPE_EXS          = static_cast<unsigned>(SVGLengthType::Exs),
        SVG_LENGTHTYPE_PX           = static_cast<unsigned>(SVGLengthType::Pixels),
        SVG_LENGTHTYPE_CM           = static_cast<unsigned>(SVGLengthType::Centimeters),
        SVG_LENGTHTYPE_MM           = static_cast<unsigned>(SVGLengthType::Millimeters),
        SVG_LENGTHTYPE_IN           = static_cast<unsigned>(SVGLengthType::Inches),
        SVG_LENGTHTYPE_PT           = static_cast<unsigned>(SVGLengthType::Points),
        SVG_LENGTHTYPE_PC           = static_cast<unsigned>(SVGLengthType::Picas)
    };

    static Ref<SVGLength> create()
    {
        return adoptRef(*new SVGLength());
    }

    static Ref<SVGLength> create(const SVGLengthValue& value)
    {
        return adoptRef(*new SVGLength(value));
    }

    static Ref<SVGLength> create(SVGPropertyOwner* owner, SVGPropertyAccess access, const SVGLengthValue& value = { })
    {
        return adoptRef(*new SVGLength(owner, access, value));
    }

    template<typename T>
    static ExceptionOr<Ref<SVGLength>> create(ExceptionOr<T>&& value)
    {
        if (value.hasException())
            return value.releaseException();
        return adoptRef(*new SVGLength(value.releaseReturnValue()));
    }

    Ref<SVGLength> clone() const
    {
        return SVGLength::create(m_value);
    }

    unsigned short unitType()  const
    {
        // Per spec: https://svgwg.org/svg2-draft/types.html#__svg__SVGLength__SVG_LENGTHTYPE_UNKNOWN
        if (m_value.lengthType() > SVGLengthType::Picas)
            return 0;
        return static_cast<unsigned>(m_value.lengthType());
    }

    ExceptionOr<float> valueForBindings();

    ExceptionOr<void> setValueForBindings(float);
    
    float valueInSpecifiedUnits()
    {
        return m_value.valueInSpecifiedUnits();
    }

    ExceptionOr<void> setValueInSpecifiedUnits(float valueInSpecifiedUnits)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        m_value.setValueInSpecifiedUnits(valueInSpecifiedUnits);
        commitChange();
        return { };
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

    ExceptionOr<void> newValueSpecifiedUnits(unsigned short unitType, float valueInSpecifiedUnits)
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };

        if (unitType == SVG_LENGTHTYPE_UNKNOWN || unitType > SVG_LENGTHTYPE_PC)
            return Exception { ExceptionCode::NotSupportedError };

        m_value = { valueInSpecifiedUnits, static_cast<SVGLengthType>(unitType), m_value.lengthMode() };
        commitChange();
        return { };
    }
    
    ExceptionOr<void> convertToSpecifiedUnits(unsigned short unitType);
    
    String valueAsString() const override
    {
        return m_value.valueAsString();
    }
};

} // namespace WebCore
