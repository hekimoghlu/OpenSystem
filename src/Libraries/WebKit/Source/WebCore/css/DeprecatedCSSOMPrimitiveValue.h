/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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

#include "CSSParserIdioms.h"
#include "CSSPrimitiveValue.h"
#include "DeprecatedCSSOMValue.h"

namespace WebCore {

class DeprecatedCSSOMCounter;
class DeprecatedCSSOMRGBColor;
class DeprecatedCSSOMRect;
    
class DeprecatedCSSOMPrimitiveValue : public DeprecatedCSSOMValue {
public:
    // Only expose what's in the IDL file.
    enum UnitType {
        CSS_UNKNOWN = 0,
        CSS_NUMBER = 1,
        CSS_PERCENTAGE = 2,
        CSS_EMS = 3,
        CSS_EXS = 4,
        CSS_PX = 5,
        CSS_CM = 6,
        CSS_MM = 7,
        CSS_IN = 8,
        CSS_PT = 9,
        CSS_PC = 10,
        CSS_DEG = 11,
        CSS_RAD = 12,
        CSS_GRAD = 13,
        CSS_MS = 14,
        CSS_S = 15,
        CSS_HZ = 16,
        CSS_KHZ = 17,
        CSS_DIMENSION = 18,
        CSS_STRING = 19,
        CSS_URI = 20,
        CSS_IDENT = 21,
        CSS_ATTR = 22,
        CSS_COUNTER = 23,
        CSS_RECT = 24,
        CSS_RGBCOLOR = 25
        // Do not add new units here; this is deprecated and we shouldn't expose anything not in DOM Level 2 Style.
    };

    static Ref<DeprecatedCSSOMPrimitiveValue> create(const CSSValue& value, CSSStyleDeclaration& owner)
    {
        return adoptRef(*new DeprecatedCSSOMPrimitiveValue(value, owner));
    }

    bool equals(const DeprecatedCSSOMPrimitiveValue& other) const { return m_value->equals(other.m_value); }
    String cssText() const { return m_value->cssText(); }
    
    WEBCORE_EXPORT unsigned short primitiveType() const;
    WEBCORE_EXPORT ExceptionOr<float> getFloatValue(unsigned short unitType) const;
    WEBCORE_EXPORT ExceptionOr<String> getStringValue() const;
    WEBCORE_EXPORT ExceptionOr<Ref<DeprecatedCSSOMCounter>> getCounterValue() const;
    WEBCORE_EXPORT ExceptionOr<Ref<DeprecatedCSSOMRect>> getRectValue() const;
    WEBCORE_EXPORT ExceptionOr<Ref<DeprecatedCSSOMRGBColor>> getRGBColorValue() const;

    static ExceptionOr<void> setFloatValue(unsigned short, double) { return Exception { ExceptionCode::NoModificationAllowedError }; }
    static ExceptionOr<void> setStringValue(unsigned short, const String&) { return Exception { ExceptionCode::NoModificationAllowedError }; }

    bool isCSSWideKeyword() const { return WebCore::isCSSWideKeyword(valueID(m_value.get())); }
    static unsigned short cssValueType() { return CSS_PRIMITIVE_VALUE; }

private:
    DeprecatedCSSOMPrimitiveValue(const CSSValue& value, CSSStyleDeclaration& owner)
        : DeprecatedCSSOMValue(ClassType::Primitive, owner)
        , m_value(value)
    {
    }

    Ref<const CSSValue> m_value;
};
    
} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSSOM_VALUE(DeprecatedCSSOMPrimitiveValue, isPrimitiveValue())
