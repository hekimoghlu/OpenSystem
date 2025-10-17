/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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
#include "StyleProperties.h"
#include "StyleRule.h"
#include <wtf/text/AtomString.h>

namespace WebCore {

enum class ViewTransitionNavigation : bool {
    Auto,
    None,
};

class StyleRuleViewTransition final : public StyleRuleBase {
public:
    static Ref<StyleRuleViewTransition> create(Ref<StyleProperties>&&);
    ~StyleRuleViewTransition();

    Ref<StyleRuleViewTransition> copy() const { return adoptRef(*new StyleRuleViewTransition(*this)); }

    std::optional<ViewTransitionNavigation> navigation() const { return m_navigation; }
    ViewTransitionNavigation computedNavigation() const { return navigation().value_or(ViewTransitionNavigation::None); }
    Vector<AtomString> types() const { return m_types; }

private:
    explicit StyleRuleViewTransition(Ref<StyleProperties>&&);
    StyleRuleViewTransition(const StyleRuleViewTransition&) = default;

    std::optional<ViewTransitionNavigation> m_navigation;
    Vector<AtomString> m_types;
};

class CSSViewTransitionRule final : public CSSRule {
public:
    using ViewTransitionNavigation = WebCore::ViewTransitionNavigation;

    static Ref<CSSViewTransitionRule> create(StyleRuleViewTransition&, CSSStyleSheet*);
    virtual ~CSSViewTransitionRule();

    String cssText() const final;
    void reattach(StyleRuleBase&) final;
    StyleRuleType styleRuleType() const final { return StyleRuleType::ViewTransition; }

    AtomString navigation() const;
    Vector<AtomString> types() const { return m_viewTransitionRule->types(); }

private:
    CSSViewTransitionRule(StyleRuleViewTransition&, CSSStyleSheet* parent);

    Ref<StyleRuleViewTransition> m_viewTransitionRule;
};


} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_RULE(CSSViewTransitionRule, StyleRuleType::ViewTransition)

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::StyleRuleViewTransition)
static bool isType(const WebCore::StyleRuleBase& rule) { return rule.isViewTransitionRule(); }
SPECIALIZE_TYPE_TRAITS_END()
