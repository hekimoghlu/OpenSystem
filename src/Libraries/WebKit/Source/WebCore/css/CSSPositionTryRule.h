/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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
#include "StylePropertiesInlines.h"
#include "StyleRule.h"
#include <wtf/TypeCasts.h>

namespace WebCore {

class CSSStyleDeclaration;
class StyleRuleCSSStyleDeclaration;

class StyleRulePositionTry final : public StyleRuleBase {
public:
    static Ref<StyleRulePositionTry> create(AtomString&& name, Ref<StyleProperties>&&);

    Ref<StyleRulePositionTry> copy() const { return adoptRef(*new StyleRulePositionTry(*this)); }

    AtomString name() const { return m_name; }

    Ref<StyleProperties> protectedProperties() const { return m_properties; }
    Ref<MutableStyleProperties> protectedMutableProperties();

private:
    explicit StyleRulePositionTry(AtomString&& name, Ref<StyleProperties>&&);

    AtomString m_name;
    Ref<StyleProperties> m_properties;
};

class CSSPositionTryRule final : public CSSRule {
public:
    static Ref<CSSPositionTryRule> create(StyleRulePositionTry&, CSSStyleSheet*);
    virtual ~CSSPositionTryRule();

    StyleRuleType styleRuleType() const { return StyleRuleType::PositionTry; }

    String cssText() const;
    void reattach(StyleRuleBase&);

    Ref<StyleRulePositionTry> protectedPositionTryRule() const { return m_positionTryRule; }

    WEBCORE_EXPORT AtomString name() const;
    WEBCORE_EXPORT CSSStyleDeclaration& style();

private:
    CSSPositionTryRule(StyleRulePositionTry&, CSSStyleSheet*);

    Ref<StyleRulePositionTry> m_positionTryRule;
    RefPtr<StyleRuleCSSStyleDeclaration> m_propertiesCSSOMWrapper;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::StyleRulePositionTry)
static bool isType(const WebCore::StyleRuleBase& rule) { return rule.isPositionTryRule(); }
SPECIALIZE_TYPE_TRAITS_END()
