/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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

class CSSRuleList;
class CSSStyleDeclaration;
class DeclaredStylePropertyMap;
class StylePropertyMap;
class StyleRuleCSSStyleDeclaration;
class StyleRuleNestedDeclarations;

class CSSNestedDeclarations final : public CSSRule, public CanMakeWeakPtr<CSSNestedDeclarations> {
public:
    static Ref<CSSNestedDeclarations> create(StyleRuleNestedDeclarations& rule, CSSStyleSheet* sheet) { return adoptRef(* new CSSNestedDeclarations(rule, sheet)); };

    virtual ~CSSNestedDeclarations();

    WEBCORE_EXPORT CSSStyleDeclaration& style();

private:
    CSSNestedDeclarations(StyleRuleNestedDeclarations&, CSSStyleSheet*);

    String cssText() const final;
    String cssTextInternal(StringBuilder& declarations, StringBuilder& rules) const;

    void reattach(StyleRuleBase&) final;
    StyleRuleType styleRuleType() const final { return StyleRuleType::NestedDeclarations; }

    Ref<StyleRuleNestedDeclarations> m_styleRule;
    RefPtr<StyleRuleCSSStyleDeclaration> m_propertiesCSSOMWrapper;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_RULE(CSSNestedDeclarations, StyleRuleType::NestedDeclarations)
