/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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
#include "CSSNestedDeclarations.h"

#include "DeclaredStylePropertyMap.h"
#include "MutableStyleProperties.h"
#include "PropertySetCSSStyleDeclaration.h"
#include "StyleProperties.h"
#include "StyleRule.h"

#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSNestedDeclarations::CSSNestedDeclarations(StyleRuleNestedDeclarations& rule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_styleRule(rule)
{
}

CSSNestedDeclarations::~CSSNestedDeclarations() = default;

CSSStyleDeclaration& CSSNestedDeclarations::style()
{
    if (!m_propertiesCSSOMWrapper)
        m_propertiesCSSOMWrapper = StyleRuleCSSStyleDeclaration::create(m_styleRule->mutableProperties(), *this);
    return *m_propertiesCSSOMWrapper;
}

String CSSNestedDeclarations::cssText() const
{
    return m_styleRule->properties().asText();
}

void CSSNestedDeclarations::reattach(StyleRuleBase& rule)
{
    m_styleRule = downcast<StyleRuleNestedDeclarations>(rule);

    if (m_propertiesCSSOMWrapper)
        m_propertiesCSSOMWrapper->reattach(m_styleRule->mutableProperties());
}

} // namespace WebCore
