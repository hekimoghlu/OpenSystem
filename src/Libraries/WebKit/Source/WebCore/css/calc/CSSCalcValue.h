/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 11, 2024.
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

#include "CSSCalcTree.h"
#include "CSSValue.h"
#include <wtf/Forward.h>
#include <wtf/Ref.h>

namespace WebCore {

namespace Calculation {
enum class Category : uint8_t;
}

namespace CSS {
struct Range;
}

class CSSCalcSymbolTable;
class CSSCalcSymbolsAllowed;
class CSSParserTokenRange;
class CSSToLengthConversionData;
class CalculationValue;
class RenderStyle;

struct CSSParserContext;
struct CSSPropertyParserOptions;
struct Length;
struct NoConversionDataRequiredToken;

enum CSSValueID : uint16_t;

enum class CSSUnitType : uint8_t;

class CSSCalcValue final : public CSSValue {
public:
    static RefPtr<CSSCalcValue> parse(CSSParserTokenRange&, const CSSParserContext&, Calculation::Category, CSS::Range, CSSCalcSymbolsAllowed, CSSPropertyParserOptions);

    static Ref<CSSCalcValue> create(const CalculationValue&, const RenderStyle&);
    static Ref<CSSCalcValue> create(Calculation::Category, CSS::Range, CSSCalc::Tree&&);

    ~CSSCalcValue();

    // Creates a copy of the CSSCalc::Tree with non-canonical dimensions and any symbols present in the provided symbol table resolved.
    Ref<CSSCalcValue> copySimplified(const CSSToLengthConversionData&) const;
    Ref<CSSCalcValue> copySimplified(const CSSToLengthConversionData&, const CSSCalcSymbolTable&) const;

    Calculation::Category category() const { return m_category; }
    CSS::Range range() const { return m_range; }

    CSSUnitType primitiveType() const;

    // Returns whether the CSSCalc::Tree requires `CSSToLengthConversionData` to fully resolve.
    bool requiresConversionData() const;

    double doubleValue(const CSSToLengthConversionData&) const;
    double doubleValue(const CSSToLengthConversionData&, const CSSCalcSymbolTable&) const;
    double doubleValue(NoConversionDataRequiredToken) const;
    double doubleValue(NoConversionDataRequiredToken, const CSSCalcSymbolTable&) const;
    double doubleValueDeprecated() const;

    double computeLengthPx(const CSSToLengthConversionData&) const;
    double computeLengthPx(const CSSToLengthConversionData&, const CSSCalcSymbolTable&) const;

    Ref<CalculationValue> createCalculationValue(NoConversionDataRequiredToken) const;
    Ref<CalculationValue> createCalculationValue(NoConversionDataRequiredToken, const CSSCalcSymbolTable&) const;
    Ref<CalculationValue> createCalculationValue(const CSSToLengthConversionData&) const;
    Ref<CalculationValue> createCalculationValue(const CSSToLengthConversionData&, const CSSCalcSymbolTable&) const;

    void collectComputedStyleDependencies(ComputedStyleDependencies&) const;

    String customCSSText() const;
    bool equals(const CSSCalcValue&) const;

    void dump(TextStream&) const;

    const CSSCalc::Tree& tree() const { return m_tree; }

private:
    explicit CSSCalcValue(Calculation::Category, CSS::Range, CSSCalc::Tree&&);

    double clampToPermittedRange(double) const;

    Calculation::Category m_category;
    CSS::Range m_range;
    CSSCalc::Tree m_tree;
};

TextStream& operator<<(TextStream&, const CSSCalcValue&);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSCalcValue, isCalcValue())
