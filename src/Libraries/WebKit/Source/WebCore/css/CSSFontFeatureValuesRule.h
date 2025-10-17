/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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

#include "CSSRule.h"

#include "StyleRule.h"

namespace WebCore {

class CSSFontFeatureValuesRule final : public CSSRule {
public:
    static Ref<CSSFontFeatureValuesRule> create(StyleRuleFontFeatureValues& rule, CSSStyleSheet* sheet) { return adoptRef(*new CSSFontFeatureValuesRule(rule, sheet)); }
    virtual ~CSSFontFeatureValuesRule() = default;

    const Vector<AtomString>& fontFamilies() const
    {
        return m_fontFeatureValuesRule->fontFamilies();
    }

    // Used by the CSSOM.
    AtomString fontFamily() const
    {
        StringBuilder builder;
        bool first = true;
        for (auto& family : m_fontFeatureValuesRule->fontFamilies()) {
            if (first)
                first = false;
            else
                builder.append(", "_s);
            
            builder.append(family);
        }
        return builder.toAtomString();
    }

private:
    CSSFontFeatureValuesRule(StyleRuleFontFeatureValues&, CSSStyleSheet* parent);

    StyleRuleType styleRuleType() const final { return StyleRuleType::FontFeatureValues; }
    String cssText() const final;
    void reattach(StyleRuleBase&) final;

    Ref<StyleRuleFontFeatureValues> m_fontFeatureValuesRule;
};

class CSSFontFeatureValuesBlockRule final : public CSSRule {
public:
    static Ref<CSSFontFeatureValuesBlockRule> create(StyleRuleFontFeatureValuesBlock& rule, CSSStyleSheet* sheet) { return adoptRef(*new CSSFontFeatureValuesBlockRule(rule, sheet)); }
    virtual ~CSSFontFeatureValuesBlockRule() = default;

private:
    CSSFontFeatureValuesBlockRule(StyleRuleFontFeatureValuesBlock&, CSSStyleSheet* parent);

    StyleRuleType styleRuleType() const final { return StyleRuleType::FontFeatureValuesBlock; }
    String cssText() const final;
    void reattach(StyleRuleBase&) final;

    Ref<StyleRuleFontFeatureValuesBlock> m_fontFeatureValuesBlockRule;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_RULE(CSSFontFeatureValuesRule, StyleRuleType::FontFeatureValues)
SPECIALIZE_TYPE_TRAITS_CSS_RULE(CSSFontFeatureValuesBlockRule, StyleRuleType::FontFeatureValuesBlock)
