/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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
#include "CSSFontFeatureValuesRule.h"

#include "CSSMarkup.h"

namespace WebCore {

CSSFontFeatureValuesRule::CSSFontFeatureValuesRule(StyleRuleFontFeatureValues& fontFeatureValuesRule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_fontFeatureValuesRule(fontFeatureValuesRule)
{
}

String CSSFontFeatureValuesRule::cssText() const
{
    StringBuilder builder;
    builder.append("@font-feature-values "_s);
    auto joinFontFamiliesWithSeparator = [&builder] (const auto& elements, ASCIILiteral separator) {
        bool first = true;
        for (auto element : elements) {
            if (!first)
                builder.append(separator);
            builder.append(serializeFontFamily(element));
            first = false;
        }
    };
    joinFontFamiliesWithSeparator(m_fontFeatureValuesRule->fontFamilies(), ", "_s);
    builder.append(" { "_s);
    const auto& value = m_fontFeatureValuesRule->value();
    
    auto addVariant = [&builder] (const String& variantName, const auto& tags) {
        if (!tags.isEmpty()) {
            builder.append('@', variantName, " { "_s);
            for (auto tag : tags) {
                serializeIdentifier(tag.key, builder);
                builder.append(':');
                for (auto integer : tag.value)
                    builder.append(' ', integer);
                builder.append("; "_s);
            }
            builder.append("} "_s);
        }
    };
    
    // WPT expects the order used in Servo.
    // https://searchfox.org/mozilla-central/source/servo/components/style/stylesheets/font_feature_values_rule.rs#430
    addVariant("swash"_s, value->swash());
    addVariant("stylistic"_s, value->stylistic());
    addVariant("ornaments"_s, value->ornaments());
    addVariant("annotation"_s, value->annotation());
    addVariant("character-variant"_s, value->characterVariant());
    addVariant("styleset"_s, value->styleset());
    
    builder.append('}');
    return builder.toString();
}

void CSSFontFeatureValuesRule::reattach(StyleRuleBase& rule)
{
    m_fontFeatureValuesRule = downcast<StyleRuleFontFeatureValues>(rule);
}

CSSFontFeatureValuesBlockRule::CSSFontFeatureValuesBlockRule(StyleRuleFontFeatureValuesBlock& block , CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_fontFeatureValuesBlockRule(block)
{
}

String CSSFontFeatureValuesBlockRule::cssText() const
{
    // This rule is always contained inside a FontFeatureValuesRule,
    // which is the only one we are expected to serialize to CSS.
    // We should never serialize a Block by itself.
    ASSERT_NOT_REACHED();
    return { };
}

void CSSFontFeatureValuesBlockRule::reattach(StyleRuleBase& rule)
{
    m_fontFeatureValuesBlockRule = downcast<StyleRuleFontFeatureValuesBlock>(rule);
}

} // namespace WebCore
