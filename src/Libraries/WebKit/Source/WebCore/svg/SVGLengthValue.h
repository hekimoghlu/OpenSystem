/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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

#include "ExceptionOr.h"
#include "SVGParsingError.h"
#include "SVGPropertyTraits.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class CSSPrimitiveValue;
class Element;
class SVGLengthContext;

enum class SVGLengthType : uint8_t {
    Unknown = 0,
    Number,
    Percentage,
    Ems,
    Exs,
    Pixels,
    Centimeters,
    Millimeters,
    Inches,
    Points,
    Picas,
    Lh,
    Ch
};

enum class SVGLengthMode : uint8_t {
    Width,
    Height,
    Other
};

enum class SVGLengthNegativeValuesMode : uint8_t {
    Allow,
    Forbid
};

enum class ShouldConvertNumberToPxLength : bool { No, Yes };

class SVGLengthValue {
    WTF_MAKE_TZONE_ALLOCATED(SVGLengthValue);
public:
    SVGLengthValue(SVGLengthMode = SVGLengthMode::Other, const String& valueAsString = { });
    SVGLengthValue(float valueInSpecifiedUnits, SVGLengthType, SVGLengthMode = SVGLengthMode::Other);
    SVGLengthValue(const SVGLengthContext&, float, SVGLengthType = SVGLengthType::Number, SVGLengthMode = SVGLengthMode::Other);

    static std::optional<SVGLengthValue> construct(SVGLengthMode, StringView);
    static SVGLengthValue construct(SVGLengthMode, StringView, SVGParsingError&, SVGLengthNegativeValuesMode = SVGLengthNegativeValuesMode::Allow);
    static SVGLengthValue blend(const SVGLengthValue& from, const SVGLengthValue& to, float progress);

    static SVGLengthValue fromCSSPrimitiveValue(const CSSPrimitiveValue&, const CSSToLengthConversionData&, ShouldConvertNumberToPxLength = ShouldConvertNumberToPxLength::No);
    Ref<CSSPrimitiveValue> toCSSPrimitiveValue(const Element* = nullptr) const;

    SVGLengthType lengthType() const { return m_lengthType; }
    SVGLengthMode lengthMode() const { return m_lengthMode; }

    bool isZero() const { return !m_valueInSpecifiedUnits;  }
    bool isRelative() const { return m_lengthType == SVGLengthType::Percentage || m_lengthType == SVGLengthType::Ems || m_lengthType == SVGLengthType::Exs || m_lengthType == SVGLengthType::Ch; }

    float value(const SVGLengthContext&) const;
    float valueAsPercentage() const { return m_lengthType == SVGLengthType::Percentage ? m_valueInSpecifiedUnits / 100 : m_valueInSpecifiedUnits; }
    float valueInSpecifiedUnits() const { return m_valueInSpecifiedUnits; }
    
    String valueAsString() const;
    AtomString valueAsAtomString() const;
    ExceptionOr<float> valueForBindings(const SVGLengthContext&) const;

    void setValueInSpecifiedUnits(float value) { m_valueInSpecifiedUnits = value; }
    ExceptionOr<void> setValue(const SVGLengthContext&, float);
    ExceptionOr<void> setValue(const SVGLengthContext&, float, SVGLengthType, SVGLengthMode);

    ExceptionOr<void> setValueAsString(StringView);
    ExceptionOr<void> setValueAsString(StringView, SVGLengthMode);

    ExceptionOr<void> convertToSpecifiedUnits(const SVGLengthContext&, SVGLengthType);

    friend bool operator==(SVGLengthValue, SVGLengthValue) = default;

private:
    float m_valueInSpecifiedUnits { 0 };
    SVGLengthType m_lengthType { SVGLengthType::Number };
    SVGLengthMode m_lengthMode { SVGLengthMode::Other };
};

WTF::TextStream& operator<<(WTF::TextStream&, const SVGLengthValue&);

} // namespace WebCore
