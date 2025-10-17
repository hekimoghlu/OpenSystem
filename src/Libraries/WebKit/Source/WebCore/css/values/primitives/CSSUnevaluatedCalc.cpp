/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 30, 2022.
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
#include "CSSUnevaluatedCalc.h"

#include "CSSCalcSymbolTable.h"
#include "CSSCalcSymbolsAllowed.h"
#include "CSSCalcValue.h"
#include "CSSNoConversionDataRequiredToken.h"
#include "CSSPropertyParserOptions.h"
#include "StyleBuilderState.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace CSS {

void unevaluatedCalcRef(CSSCalcValue* calc)
{
    calc->ref();
}

void unevaluatedCalcDeref(CSSCalcValue* calc)
{
    calc->deref();
}

UnevaluatedCalcBase::UnevaluatedCalcBase(CSSCalcValue& value)
    : calc { value }
{
}

UnevaluatedCalcBase::UnevaluatedCalcBase(Ref<CSSCalcValue>&& value)
    : calc { WTFMove(value) }
{
}

UnevaluatedCalcBase::UnevaluatedCalcBase(const UnevaluatedCalcBase&) = default;
UnevaluatedCalcBase::UnevaluatedCalcBase(UnevaluatedCalcBase&&) = default;
UnevaluatedCalcBase& UnevaluatedCalcBase::operator=(const UnevaluatedCalcBase&) = default;
UnevaluatedCalcBase& UnevaluatedCalcBase::operator=(UnevaluatedCalcBase&&) = default;

UnevaluatedCalcBase::~UnevaluatedCalcBase() = default;

Ref<CSSCalcValue> UnevaluatedCalcBase::protectedCalc() const
{
    return calc;
}

CSSCalcValue& UnevaluatedCalcBase::leakRef()
{
    return calc.leakRef();
}

bool UnevaluatedCalcBase::equal(const UnevaluatedCalcBase& other) const
{
    return protectedCalc()->equals(other.calc.get());
}

bool UnevaluatedCalcBase::requiresConversionData() const
{
    return protectedCalc()->requiresConversionData();
}

void UnevaluatedCalcBase::serializationForCSS(StringBuilder& builder) const
{
    builder.append(protectedCalc()->customCSSText());
}

void UnevaluatedCalcBase::collectComputedStyleDependencies(ComputedStyleDependencies& dependencies) const
{
    protectedCalc()->collectComputedStyleDependencies(dependencies);
}

IterationStatus UnevaluatedCalcBase::visitChildren(const Function<IterationStatus(CSSValue&)>& func) const
{
    return func(calc);
}

UnevaluatedCalcBase UnevaluatedCalcBase::simplifyBase(const CSSToLengthConversionData& conversionData, const CSSCalcSymbolTable& symbolTable) const
{
    return UnevaluatedCalcBase { protectedCalc()->copySimplified(conversionData, symbolTable) };
}

double UnevaluatedCalcBase::evaluate(Calculation::Category category, const Style::BuilderState& state) const
{
    return evaluate(category, state.cssToLengthConversionData(), { });
}

double UnevaluatedCalcBase::evaluate(Calculation::Category category, const Style::BuilderState& state, const CSSCalcSymbolTable& symbolTable) const
{
    return evaluate(category, state.cssToLengthConversionData(), symbolTable);
}

double UnevaluatedCalcBase::evaluate(Calculation::Category category, const CSSToLengthConversionData& conversionData) const
{
    return evaluate(category, conversionData, { });
}

double UnevaluatedCalcBase::evaluate(Calculation::Category category, const CSSToLengthConversionData& conversionData, const CSSCalcSymbolTable& symbolTable) const
{
    ASSERT_UNUSED(category, protectedCalc()->category() == category);
    return protectedCalc()->doubleValue(conversionData, symbolTable);
}

double UnevaluatedCalcBase::evaluate(Calculation::Category category, NoConversionDataRequiredToken token) const
{
    return evaluate(category, token, { });
}

double UnevaluatedCalcBase::evaluate(Calculation::Category category, NoConversionDataRequiredToken token, const CSSCalcSymbolTable& symbolTable) const
{
    ASSERT_UNUSED(category, protectedCalc()->category() == category);
    return protectedCalc()->doubleValue(token, symbolTable);
}

} // namespace CSS
} // namespace WebCore
