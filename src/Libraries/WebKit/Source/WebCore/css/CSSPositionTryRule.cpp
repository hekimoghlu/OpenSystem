/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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
#include "CSSPositionTryRule.h"

#include "PropertySetCSSStyleDeclaration.h"

namespace WebCore {

Ref<StyleRulePositionTry> StyleRulePositionTry::create(AtomString&& name, Ref<StyleProperties>&& properties)
{
    return adoptRef(*new StyleRulePositionTry(WTFMove(name), WTFMove(properties)));
}

StyleRulePositionTry::StyleRulePositionTry(AtomString&& name, Ref<StyleProperties>&& properties)
    : StyleRuleBase(StyleRuleType::PositionTry)
    , m_name(name)
    , m_properties(properties)
{
}

Ref<MutableStyleProperties> StyleRulePositionTry::protectedMutableProperties()
{
    auto propertiesRef = protectedProperties();

    if (!is<MutableStyleProperties>(propertiesRef))
        m_properties = propertiesRef->mutableCopy();

    return downcast<MutableStyleProperties>(m_properties.get());
}

Ref<CSSPositionTryRule> CSSPositionTryRule::create(StyleRulePositionTry& rule, CSSStyleSheet* parent)
{
    return adoptRef(*new CSSPositionTryRule(rule, parent));
}

CSSPositionTryRule::CSSPositionTryRule(StyleRulePositionTry& rule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_positionTryRule(rule)
    , m_propertiesCSSOMWrapper(nullptr)
{
}

CSSPositionTryRule::~CSSPositionTryRule() = default;

String CSSPositionTryRule::cssText() const
{
    StringBuilder builder;
    builder.append("@position-try "_s, name(), " {"_s);

    auto propertiesRef = m_positionTryRule->protectedProperties();

    if (auto declarations = propertiesRef->asText(); !declarations.isEmpty())
        builder.append(' ', declarations, ' ');
    else
        builder.append(' ');

    builder.append('}');

    return builder.toString();
}

void CSSPositionTryRule::reattach(StyleRuleBase& rule)
{
    m_positionTryRule = downcast<StyleRulePositionTry>(rule);
}

AtomString CSSPositionTryRule::name() const
{
    return m_positionTryRule->name();
}

CSSStyleDeclaration& CSSPositionTryRule::style()
{
    Ref mutablePropertiesRef = protectedPositionTryRule()->protectedMutableProperties();

    if (!m_propertiesCSSOMWrapper)
        m_propertiesCSSOMWrapper = StyleRuleCSSStyleDeclaration::create(mutablePropertiesRef.get(), *this);

    return *m_propertiesCSSOMWrapper;
}

} // namespace WebCore
