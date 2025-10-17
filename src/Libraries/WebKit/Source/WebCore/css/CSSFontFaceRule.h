/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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

namespace WebCore {

class CSSStyleDeclaration;
class StyleRuleFontFace;
class StyleRuleCSSStyleDeclaration;

class CSSFontFaceRule final : public CSSRule {
public:
    static Ref<CSSFontFaceRule> create(StyleRuleFontFace& rule, CSSStyleSheet* sheet) { return adoptRef(*new CSSFontFaceRule(rule, sheet)); }

    virtual ~CSSFontFaceRule();

    WEBCORE_EXPORT CSSStyleDeclaration& style();

private:
    CSSFontFaceRule(StyleRuleFontFace&, CSSStyleSheet* parent);

    StyleRuleType styleRuleType() const final { return StyleRuleType::FontFace; }
    String cssText() const final;
    String cssTextWithReplacementURLs(const UncheckedKeyHashMap<String, String>&, const UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>&) const final;
    String cssTextInternal(const String& declarations) const;
    void reattach(StyleRuleBase&) final;

    Ref<StyleRuleFontFace> m_fontFaceRule;
    RefPtr<StyleRuleCSSStyleDeclaration> m_propertiesCSSOMWrapper;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_RULE(CSSFontFaceRule, StyleRuleType::FontFace)
