/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
#include "CSSPageRule.h"

#include "CSSParser.h"
#include "CSSSelector.h"
#include "CSSStyleSheet.h"
#include "CommonAtomStrings.h"
#include "Document.h"
#include "PropertySetCSSStyleDeclaration.h"
#include "StyleProperties.h"
#include "StyleRule.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

CSSPageRule::CSSPageRule(StyleRulePage& pageRule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_pageRule(pageRule)
{
}

CSSPageRule::~CSSPageRule()
{
    if (m_propertiesCSSOMWrapper)
        m_propertiesCSSOMWrapper->clearParentRule();
}

CSSStyleDeclaration& CSSPageRule::style()
{
    if (!m_propertiesCSSOMWrapper)
        m_propertiesCSSOMWrapper = StyleRuleCSSStyleDeclaration::create(m_pageRule->mutableProperties(), *this);
    return *m_propertiesCSSOMWrapper;
}

String CSSPageRule::selectorText() const
{
    if (auto* selector = m_pageRule->selector()) {
        if (!selector->selectorText().isEmpty())
            return selector->selectorText();
    }
    return ""_s;
}

void CSSPageRule::setSelectorText(const String& selectorText)
{
    CSSParser parser(parserContext());
    auto* sheet = parentStyleSheet();
    auto selectorList = parser.parseSelectorList(selectorText, sheet ? &sheet->contents() : nullptr);
    if (!selectorList)
        return;

    CSSStyleSheet::RuleMutationScope mutationScope(this);

    m_pageRule->wrapperAdoptSelectorList(WTFMove(*selectorList));
}

String CSSPageRule::cssText() const
{
    auto selector = selectorText();
    auto optionalSpace = selector.isEmpty() ? ""_s : " "_s;
    if (auto declarations = m_pageRule->properties().asText(); !declarations.isEmpty())
        return makeString("@page"_s, optionalSpace, selector, " { "_s, declarations, " }"_s);
    return makeString("@page"_s, optionalSpace, selector, " { }"_s);
}

void CSSPageRule::reattach(StyleRuleBase& rule)
{
    m_pageRule = downcast<StyleRulePage>(rule);
    if (m_propertiesCSSOMWrapper)
        m_propertiesCSSOMWrapper->reattach(m_pageRule.get().mutableProperties());
}

} // namespace WebCore
