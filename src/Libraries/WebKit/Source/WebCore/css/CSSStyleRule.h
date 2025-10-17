/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
#include <wtf/WeakPtr.h>

namespace WebCore {

class CSSRuleList;
class CSSStyleDeclaration;
class DeclaredStylePropertyMap;
class StylePropertyMap;
class StyleRuleCSSStyleDeclaration;
class StyleRule;
class StyleRuleWithNesting;
class StyleRuleCSSStyleDeclaration;

class CSSStyleRule final : public CSSRule, public CanMakeWeakPtr<CSSStyleRule> {
public:
    static Ref<CSSStyleRule> create(StyleRule& rule, CSSStyleSheet* sheet) { return adoptRef(*new CSSStyleRule(rule, sheet)); }
    static Ref<CSSStyleRule> create(StyleRuleWithNesting& rule, CSSStyleSheet* sheet) { return adoptRef(* new CSSStyleRule(rule, sheet)); };

    virtual ~CSSStyleRule();

    WEBCORE_EXPORT String selectorText() const;
    WEBCORE_EXPORT void setSelectorText(const String&);

    WEBCORE_EXPORT CSSStyleDeclaration& style();

    // FIXME: Not CSSOM. Remove.
    StyleRule& styleRule() const { return m_styleRule.get(); }

    WEBCORE_EXPORT CSSRuleList& cssRules() const;
    WEBCORE_EXPORT ExceptionOr<unsigned> insertRule(const String& rule, unsigned index);
    WEBCORE_EXPORT ExceptionOr<void> deleteRule(unsigned index);
    unsigned length() const;
    CSSRule* item(unsigned index) const;

    StylePropertyMap& styleMap();

private:
    CSSStyleRule(StyleRule&, CSSStyleSheet*);
    CSSStyleRule(StyleRuleWithNesting&, CSSStyleSheet*);

    StyleRuleType styleRuleType() const final { return StyleRuleType::Style; }
    String cssText() const final;
    String cssTextWithReplacementURLs(const UncheckedKeyHashMap<String, String>&, const UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>&) const final;
    String cssTextInternal(StringBuilder& declarations, StringBuilder& rules) const;
    void reattach(StyleRuleBase&) final;
    void getChildStyleSheets(UncheckedKeyHashSet<RefPtr<CSSStyleSheet>>&) final;

    String generateSelectorText() const;
    Vector<Ref<StyleRuleBase>> nestedRules() const;
    void cssTextForRules(StringBuilder& rules) const;
    void cssTextForRulesWithReplacementURLs(StringBuilder& rules, const UncheckedKeyHashMap<String, String>&, const UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>&) const;

    Ref<StyleRule> m_styleRule;
    Ref<DeclaredStylePropertyMap> m_styleMap;
    RefPtr<StyleRuleCSSStyleDeclaration> m_propertiesCSSOMWrapper;

    mutable Vector<RefPtr<CSSRule>> m_childRuleCSSOMWrappers;
    mutable std::unique_ptr<CSSRuleList> m_ruleListCSSOMWrapper;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_RULE(CSSStyleRule, StyleRuleType::Style)
