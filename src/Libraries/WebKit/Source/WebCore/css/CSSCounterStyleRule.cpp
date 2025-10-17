/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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
#include "CSSCounterStyleRule.h"

#include "CSSCounterStyleDescriptors.h"
#include "CSSPropertyParser.h"
#include "CSSPropertyParserConsumer+CounterStyles.h"
#include "CSSStyleSheet.h"
#include "CSSTokenizer.h"
#include "CSSValuePair.h"
#include "MutableStyleProperties.h"
#include "StyleProperties.h"
#include "StylePropertiesInlines.h"
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

StyleRuleCounterStyle::StyleRuleCounterStyle(const AtomString& name, CSSCounterStyleDescriptors&& descriptors)
    : StyleRuleBase(StyleRuleType::CounterStyle)
    , m_name(name)
    , m_descriptors(WTFMove(descriptors))
{
}

Ref<StyleRuleCounterStyle> StyleRuleCounterStyle::create(const AtomString& name, CSSCounterStyleDescriptors&& descriptors)
{
    return adoptRef(*new StyleRuleCounterStyle(name, WTFMove(descriptors)));
}

CSSCounterStyleDescriptors::System toCounterStyleSystemEnum(const CSSValue* system)
{
    if (!system)
        return CSSCounterStyleDescriptors::System::Symbolic;

    ASSERT(system->isValueID() || system->isPair());
    CSSValueID systemKeyword = CSSValueInvalid;
    if (system->isValueID())
        systemKeyword = system->valueID();
    else if (system->isPair()) {
        // This value must be `fixed` or `extends`, both of which can or must have an additional component.
        systemKeyword = system->first().valueID();
    }
    switch (systemKeyword) {
    case CSSValueCyclic:
        return CSSCounterStyleDescriptors::System::Cyclic;
    case CSSValueFixed:
        return CSSCounterStyleDescriptors::System::Fixed;
    case CSSValueSymbolic:
        return CSSCounterStyleDescriptors::System::Symbolic;
    case CSSValueAlphabetic:
        return CSSCounterStyleDescriptors::System::Alphabetic;
    case CSSValueNumeric:
        return CSSCounterStyleDescriptors::System::Numeric;
    case CSSValueAdditive:
        return CSSCounterStyleDescriptors::System::Additive;
    case CSSValueInternalDisclosureClosed:
        return CSSCounterStyleDescriptors::System::DisclosureClosed;
    case CSSValueInternalDisclosureOpen:
        return CSSCounterStyleDescriptors::System::DisclosureOpen;
    case CSSValueInternalSimplifiedChineseInformal:
        return CSSCounterStyleDescriptors::System::SimplifiedChineseInformal;
    case CSSValueInternalSimplifiedChineseFormal:
        return CSSCounterStyleDescriptors::System::SimplifiedChineseFormal;
    case CSSValueInternalTraditionalChineseInformal:
        return CSSCounterStyleDescriptors::System::TraditionalChineseInformal;
    case CSSValueInternalTraditionalChineseFormal:
        return CSSCounterStyleDescriptors::System::TraditionalChineseFormal;
    case CSSValueInternalEthiopicNumeric:
        return CSSCounterStyleDescriptors::System::EthiopicNumeric;
    case CSSValueExtends:
        return CSSCounterStyleDescriptors::System::Extends;
    default:
        ASSERT_NOT_REACHED();
        return CSSCounterStyleDescriptors::System::Symbolic;
    }
}

StyleRuleCounterStyle::~StyleRuleCounterStyle() = default;

Ref<CSSCounterStyleRule> CSSCounterStyleRule::create(StyleRuleCounterStyle& rule, CSSStyleSheet* sheet)
{
    return adoptRef(*new CSSCounterStyleRule(rule, sheet));
}

CSSCounterStyleRule::CSSCounterStyleRule(StyleRuleCounterStyle& counterStyleRule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_counterStyleRule(counterStyleRule)
{
}

CSSCounterStyleRule::~CSSCounterStyleRule() = default;

String CSSCounterStyleRule::cssText() const
{
    String systemText = system();
    const auto systemPrefix = systemText.isEmpty() ? ""_s : " system: "_s;
    const auto systemSuffix = systemText.isEmpty() ? ""_s : ";"_s;

    String symbolsText = symbols();
    const auto symbolsPrefix = symbolsText.isEmpty() ? ""_s : " symbols: "_s;
    const auto symbolsSuffix = symbolsText.isEmpty() ? ""_s : ";"_s;

    String additiveSymbolsText = additiveSymbols();
    const auto additiveSymbolsPrefix = additiveSymbolsText.isEmpty() ? ""_s : " additive-symbols: "_s;
    const auto additiveSymbolsSuffix = additiveSymbolsText.isEmpty() ? ""_s : ";"_s;

    String negativeText = negative();
    const auto negativePrefix = negativeText.isEmpty() ? ""_s : " negative: "_s;
    const auto negativeSuffix = negativeText.isEmpty() ? ""_s : ";"_s;

    String prefixText = prefix();
    const auto prefixTextPrefix = prefixText.isEmpty() ? ""_s : " prefix: "_s;
    const auto prefixTextSuffix = prefixText.isEmpty() ? ""_s : ";"_s;

    String suffixText = suffix();
    const auto suffixTextPrefix = suffixText.isEmpty() ? ""_s : " suffix: "_s;
    const auto suffixTextSuffix = suffixText.isEmpty() ? ""_s : ";"_s;

    String padText = pad();
    const auto padPrefix = padText.isEmpty() ? ""_s : " pad: "_s;
    const auto padSuffix = padText.isEmpty() ? ""_s : ";"_s;

    String rangeText = range();
    const auto rangePrefix = rangeText.isEmpty() ? ""_s : " range: "_s;
    const auto rangeSuffix = rangeText.isEmpty() ? ""_s : ";"_s;

    String fallbackText = fallback();
    const auto fallbackPrefix = fallbackText.isEmpty() ? ""_s : " fallback: "_s;
    const auto fallbackSuffix = fallbackText.isEmpty() ? ""_s : ";"_s;

    String speakAsText = speakAs();
    const auto speakAsPrefix = speakAsText.isEmpty() ? ""_s : " speak-as: "_s;
    const auto speakAsSuffix = speakAsText.isEmpty() ? ""_s : ";"_s;

    return makeString("@counter-style "_s, name(), " {"_s,
        systemPrefix, systemText, systemSuffix,
        symbolsPrefix, symbolsText, symbolsSuffix,
        additiveSymbolsPrefix, additiveSymbolsText, additiveSymbolsSuffix,
        negativePrefix, negativeText, negativeSuffix,
        prefixTextPrefix, prefixText, prefixTextSuffix,
        suffixTextPrefix, suffixText, suffixTextSuffix,
        padPrefix, padText, padSuffix,
        rangePrefix, rangeText, rangeSuffix,
        fallbackPrefix, fallbackText, fallbackSuffix,
        speakAsPrefix, speakAsText, speakAsSuffix,
    " }"_s);
}

void CSSCounterStyleRule::reattach(StyleRuleBase& rule)
{
    m_counterStyleRule = downcast<StyleRuleCounterStyle>(rule);
}

RefPtr<CSSValue> CSSCounterStyleRule::cssValueFromText(CSSPropertyID propertyID, const String& valueText)
{
    auto tokenizer = CSSTokenizer(valueText);
    auto tokenRange = tokenizer.tokenRange();
    return CSSPropertyParser::parseCounterStyleDescriptor(propertyID, tokenRange, parserContext());
}

// https://drafts.csswg.org/css-counter-styles-3/#dom-csscounterstylerule-name
void CSSCounterStyleRule::setName(const String& text)
{
    auto tokenizer = CSSTokenizer(text);
    auto tokenRange = tokenizer.tokenRange();
    auto name = CSSPropertyParserHelpers::consumeCounterStyleNameInPrelude(tokenRange);
    if (!name)
        return;
    CSSStyleSheet::RuleMutationScope mutationScope(this);
    mutableDescriptors().setName(WTFMove(name));
}

void CSSCounterStyleRule::setSystem(const String& text)
{
    auto systemValue = cssValueFromText(CSSPropertySystem, text);
    if (!systemValue)
        return;
    auto system = toCounterStyleSystemEnum(systemValue.get());
    // If the attribute being set is `system`, and the new value would change the algorithm used, do nothing
    // and abort these steps.
    // (It's okay to change an aspect of the algorithm, like the first symbol value of a `fixed` system.)
    // https://www.w3.org/TR/css-counter-styles-3/#the-csscounterstylerule-interface
    auto systemData = extractSystemDataFromCSSValue(WTFMove(systemValue), system);
    CSSStyleSheet::RuleMutationScope mutationScope(this);
    mutableDescriptors().setSystemData(WTFMove(systemData));
}

void CSSCounterStyleRule::setNegative(const String& text)
{
    auto newValue = cssValueFromText(CSSPropertyNegative, text);
    if (!newValue)
        return;
    CSSStyleSheet::RuleMutationScope mutationScope(this);
    mutableDescriptors().setNegative(negativeSymbolsFromCSSValue(newValue.releaseNonNull()));
}

void CSSCounterStyleRule::setPrefix(const String& text)
{
    auto newValue = cssValueFromText(CSSPropertyPrefix, text);
    if (!newValue)
        return;
    CSSStyleSheet::RuleMutationScope mutationScope(this);
    mutableDescriptors().setPrefix(symbolFromCSSValue(WTFMove(newValue)));
}

void CSSCounterStyleRule::setSuffix(const String& text)
{
    auto newValue = cssValueFromText(CSSPropertySuffix, text);
    if (!newValue)
        return;
    CSSStyleSheet::RuleMutationScope mutationScope(this);
    mutableDescriptors().setSuffix(symbolFromCSSValue(WTFMove(newValue)));
}

void CSSCounterStyleRule::setRange(const String& text)
{
    auto newValue = cssValueFromText(CSSPropertyRange, text);
    if (!newValue)
        return;
    CSSStyleSheet::RuleMutationScope mutationScope(this);
    mutableDescriptors().setRanges(rangeFromCSSValue(newValue.releaseNonNull()));
}

void CSSCounterStyleRule::setPad(const String& text)
{
    auto newValue = cssValueFromText(CSSPropertyPad, text);
    if (!newValue)
        return;
    CSSStyleSheet::RuleMutationScope mutationScope(this);
    mutableDescriptors().setPad(padFromCSSValue(newValue.releaseNonNull()));
}

void CSSCounterStyleRule::setFallback(const String& text)
{
    auto newValue = cssValueFromText(CSSPropertyFallback, text);
    if (!newValue)
        return;
    CSSStyleSheet::RuleMutationScope mutationScope(this);
    mutableDescriptors().setFallbackName(fallbackNameFromCSSValue(newValue.releaseNonNull()));
}

void CSSCounterStyleRule::setSymbols(const String& text)
{
    auto newValue = cssValueFromText(CSSPropertySymbols, text);
    if (!newValue)
        return;
    CSSStyleSheet::RuleMutationScope mutationScope(this);
    mutableDescriptors().setSymbols(symbolsFromCSSValue(newValue.releaseNonNull()));
}

void CSSCounterStyleRule::setAdditiveSymbols(const String& text)
{
    auto newValue = cssValueFromText(CSSPropertyAdditiveSymbols, text);
    if (!newValue)
        return;
    CSSStyleSheet::RuleMutationScope mutationScope(this);
    mutableDescriptors().setAdditiveSymbols(additiveSymbolsFromCSSValue(newValue.releaseNonNull()));
}

void CSSCounterStyleRule::setSpeakAs(const String&)
{
    // FIXME: @counter-style speak-as not supported (rdar://103019111).
}

} // namespace WebCore
