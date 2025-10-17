/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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
#include <memory>
#include <wtf/Vector.h>

namespace WebCore {

class CSSRuleList;
class StyleRuleGroup;

class CSSGroupingRule : public CSSRule {
public:
    virtual ~CSSGroupingRule();

    WEBCORE_EXPORT CSSRuleList& cssRules() const;
    WEBCORE_EXPORT ExceptionOr<unsigned> insertRule(const String& rule, unsigned index);
    WEBCORE_EXPORT ExceptionOr<void> deleteRule(unsigned index);
    unsigned length() const;
    CSSRule* item(unsigned index) const;

protected:
    CSSGroupingRule(StyleRuleGroup&, CSSStyleSheet* parent);
    const StyleRuleGroup& groupRule() const { return m_groupRule; }
    StyleRuleGroup& groupRule() { return m_groupRule; }
    void reattach(StyleRuleBase&) override;
    void appendCSSTextForItems(StringBuilder&) const;
    void appendCSSTextWithReplacementURLsForItems(StringBuilder&, const UncheckedKeyHashMap<String, String>&, const UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>&) const;
    RefPtr<StyleRuleWithNesting> prepareChildStyleRuleForNesting(StyleRule&) override;

private:
    bool isGroupingRule() const final { return true; }
    void appendCSSTextForItemsInternal(StringBuilder&, StringBuilder&) const;
    void cssTextForRules(StringBuilder&) const;
    void cssTextForRulesWithReplacementURLs(StringBuilder&, const UncheckedKeyHashMap<String, String>&, const UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>&) const;

    Ref<StyleRuleGroup> m_groupRule;
    mutable Vector<RefPtr<CSSRule>> m_childRuleCSSOMWrappers;
    mutable std::unique_ptr<CSSRuleList> m_ruleListCSSOMWrapper;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSGroupingRule)
    static bool isType(const WebCore::CSSRule& rule) { return rule.isGroupingRule(); }
SPECIALIZE_TYPE_TRAITS_END()
