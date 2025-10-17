/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
class CSSStyleSheet;
class StyleRulePage;
class StyleRuleCSSStyleDeclaration;

class CSSPageRule final : public CSSRule {
public:
    static Ref<CSSPageRule> create(StyleRulePage& rule, CSSStyleSheet* sheet) { return adoptRef(*new CSSPageRule(rule, sheet)); }

    virtual ~CSSPageRule();

    WEBCORE_EXPORT CSSStyleDeclaration& style();

    WEBCORE_EXPORT String selectorText() const;
    WEBCORE_EXPORT void setSelectorText(const String&);

private:
    CSSPageRule(StyleRulePage&, CSSStyleSheet*);

    StyleRuleType styleRuleType() const final { return StyleRuleType::Page; }
    String cssText() const final;
    void reattach(StyleRuleBase&) final;

    Ref<StyleRulePage> m_pageRule;
    mutable RefPtr<StyleRuleCSSStyleDeclaration> m_propertiesCSSOMWrapper;
};

} // namespace WebCore
