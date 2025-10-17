/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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
#include "config.h"
#include "SVGAngleValue.h"

#include "SVGParserUtilities.h"
#include <wtf/MathExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/FastCharacterComparison.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringParsingBuffer.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAngleValue);

float SVGAngleValue::value() const
{
    switch (m_unitType) {
    case SVG_ANGLETYPE_GRAD:
        return grad2deg(m_valueInSpecifiedUnits);
    case SVG_ANGLETYPE_RAD:
        return rad2deg(m_valueInSpecifiedUnits);
    case SVG_ANGLETYPE_TURN:
        return turn2deg(m_valueInSpecifiedUnits);
    case SVG_ANGLETYPE_UNSPECIFIED:
    case SVG_ANGLETYPE_UNKNOWN:
    case SVG_ANGLETYPE_DEG:
        return m_valueInSpecifiedUnits;
    }
    ASSERT_NOT_REACHED();
    return 0;
}

void SVGAngleValue::setValue(float value)
{
    switch (m_unitType) {
    case SVG_ANGLETYPE_GRAD:
        m_valueInSpecifiedUnits = deg2grad(value);
        return;
    case SVG_ANGLETYPE_RAD:
        m_valueInSpecifiedUnits = deg2rad(value);
        return;
    case SVG_ANGLETYPE_TURN:
        m_valueInSpecifiedUnits = deg2turn(value);
        return;
    case SVG_ANGLETYPE_UNSPECIFIED:
    case SVG_ANGLETYPE_UNKNOWN:
    case SVG_ANGLETYPE_DEG:
        m_valueInSpecifiedUnits = value;
        return;
    }
    ASSERT_NOT_REACHED();
}

String SVGAngleValue::valueAsString() const
{
    switch (m_unitType) {
    case SVG_ANGLETYPE_DEG:
        return makeString(m_valueInSpecifiedUnits, "deg"_s);
    case SVG_ANGLETYPE_RAD:
        return makeString(m_valueInSpecifiedUnits, "rad"_s);
    case SVG_ANGLETYPE_TURN:
        return makeString(m_valueInSpecifiedUnits, "turn"_s);
    case SVG_ANGLETYPE_GRAD:
        return makeString(m_valueInSpecifiedUnits, "grad"_s);
    case SVG_ANGLETYPE_UNSPECIFIED:
    case SVG_ANGLETYPE_UNKNOWN:
        return String::number(m_valueInSpecifiedUnits);
    }

    ASSERT_NOT_REACHED();
    return String();
}

template<typename CharacterType> static inline SVGAngleValue::Type parseAngleType(StringParsingBuffer<CharacterType> buffer)
{
    switch (buffer.lengthRemaining()) {
    case 0:
        return SVGAngleValue::SVG_ANGLETYPE_UNSPECIFIED;
    case 3:
        if (compareCharacters(buffer.position(), 'd', 'e', 'g'))
            return SVGAngleValue::SVG_ANGLETYPE_DEG;
        if (compareCharacters(buffer.position(), 'r', 'a', 'd'))
            return SVGAngleValue::SVG_ANGLETYPE_RAD;
        break;
    case 4:
        if (compareCharacters(buffer.position(), 'g', 'r', 'a', 'd'))
            return SVGAngleValue::SVG_ANGLETYPE_GRAD;
        if (compareCharacters(buffer.position(), 't', 'u', 'r', 'n'))
            return SVGAngleValue::SVG_ANGLETYPE_TURN;
        break;
    }
    return SVGAngleValue::SVG_ANGLETYPE_UNKNOWN;
}

ExceptionOr<void> SVGAngleValue::setValueAsString(const String& value)
{
    if (value.isEmpty()) {
        m_unitType = SVG_ANGLETYPE_UNSPECIFIED;
        return { };
    }

    return readCharactersForParsing(value, [&](auto buffer) -> ExceptionOr<void> {
        auto valueInSpecifiedUnits = parseNumber(buffer, SuffixSkippingPolicy::DontSkip);
        if (!valueInSpecifiedUnits)
            return Exception { ExceptionCode::SyntaxError };

        auto unitType = parseAngleType(buffer);
        if (unitType == SVGAngleValue::SVG_ANGLETYPE_UNKNOWN)
            return Exception { ExceptionCode::SyntaxError };

        m_unitType = unitType;
        m_valueInSpecifiedUnits = *valueInSpecifiedUnits;
        return { };
    });
}

ExceptionOr<void> SVGAngleValue::newValueSpecifiedUnits(unsigned short unitType, float valueInSpecifiedUnits)
{
    if (unitType == SVG_ANGLETYPE_UNKNOWN || unitType > SVG_ANGLETYPE_GRAD)
        return Exception { ExceptionCode::NotSupportedError };

    m_unitType = static_cast<Type>(unitType);
    m_valueInSpecifiedUnits = valueInSpecifiedUnits;
    return { };
}

ExceptionOr<void> SVGAngleValue::convertToSpecifiedUnits(unsigned short unitType)
{
    if (unitType == SVG_ANGLETYPE_UNKNOWN || m_unitType == SVG_ANGLETYPE_UNKNOWN || unitType > SVG_ANGLETYPE_GRAD)
        return Exception { ExceptionCode::NotSupportedError };

    if (unitType == m_unitType)
        return { };

    switch (m_unitType) {
    case SVG_ANGLETYPE_TURN:
        switch (unitType) {
        case SVG_ANGLETYPE_GRAD:
            m_valueInSpecifiedUnits = turn2grad(m_valueInSpecifiedUnits);
            break;
        case SVG_ANGLETYPE_UNSPECIFIED:
        case SVG_ANGLETYPE_DEG:
            m_valueInSpecifiedUnits = turn2deg(m_valueInSpecifiedUnits);
            break;
        case SVG_ANGLETYPE_RAD:
            m_valueInSpecifiedUnits = deg2rad(turn2deg(m_valueInSpecifiedUnits));
            break;
        case SVG_ANGLETYPE_TURN:
        case SVG_ANGLETYPE_UNKNOWN:
            ASSERT_NOT_REACHED();
            break;
        }
        break;
    case SVG_ANGLETYPE_RAD:
        switch (unitType) {
        case SVG_ANGLETYPE_GRAD:
            m_valueInSpecifiedUnits = rad2grad(m_valueInSpecifiedUnits);
            break;
        case SVG_ANGLETYPE_UNSPECIFIED:
        case SVG_ANGLETYPE_DEG:
            m_valueInSpecifiedUnits = rad2deg(m_valueInSpecifiedUnits);
            break;
        case SVG_ANGLETYPE_TURN:
            m_valueInSpecifiedUnits = deg2turn(rad2deg(m_valueInSpecifiedUnits));
            break;
        case SVG_ANGLETYPE_RAD:
        case SVG_ANGLETYPE_UNKNOWN:
            ASSERT_NOT_REACHED();
            break;
        }
        break;
    case SVG_ANGLETYPE_GRAD:
        switch (unitType) {
        case SVG_ANGLETYPE_RAD:
            m_valueInSpecifiedUnits = grad2rad(m_valueInSpecifiedUnits);
            break;
        case SVG_ANGLETYPE_UNSPECIFIED:
        case SVG_ANGLETYPE_DEG:
            m_valueInSpecifiedUnits = grad2deg(m_valueInSpecifiedUnits);
            break;
        case SVG_ANGLETYPE_TURN:
            m_valueInSpecifiedUnits = grad2turn(m_valueInSpecifiedUnits);
            break;
        case SVG_ANGLETYPE_GRAD:
        case SVG_ANGLETYPE_UNKNOWN:
            ASSERT_NOT_REACHED();
            break;
        }
        break;
    case SVG_ANGLETYPE_UNSPECIFIED:
        // Spec: For angles, a unitless value is treated the same as if degrees were specified.
    case SVG_ANGLETYPE_DEG:
        switch (unitType) {
        case SVG_ANGLETYPE_RAD:
            m_valueInSpecifiedUnits = deg2rad(m_valueInSpecifiedUnits);
            break;
        case SVG_ANGLETYPE_GRAD:
            m_valueInSpecifiedUnits = deg2grad(m_valueInSpecifiedUnits);
            break;
        case SVG_ANGLETYPE_TURN:
            m_valueInSpecifiedUnits = deg2turn(m_valueInSpecifiedUnits);
            break;
        case SVG_ANGLETYPE_UNSPECIFIED:
        case SVG_ANGLETYPE_DEG:
            break;
        case SVG_ANGLETYPE_UNKNOWN:
            ASSERT_NOT_REACHED();
            break;
        }
        break;
    case SVG_ANGLETYPE_UNKNOWN:
        ASSERT_NOT_REACHED();
        break;
    }

    m_unitType = static_cast<Type>(unitType);

    return { };
}

}
