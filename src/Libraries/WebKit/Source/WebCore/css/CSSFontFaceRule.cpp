/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#include "CSSFontFaceRule.h"

#include "MutableStyleProperties.h"
#include "PropertySetCSSStyleDeclaration.h"
#include "StyleProperties.h"
#include "StyleRule.h"
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSFontFaceRule::CSSFontFaceRule(StyleRuleFontFace& fontFaceRule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_fontFaceRule(fontFaceRule)
{
}

CSSFontFaceRule::~CSSFontFaceRule()
{
    if (m_propertiesCSSOMWrapper)
        m_propertiesCSSOMWrapper->clearParentRule();
}

CSSStyleDeclaration& CSSFontFaceRule::style()
{
    if (!m_propertiesCSSOMWrapper)
        m_propertiesCSSOMWrapper = StyleRuleCSSStyleDeclaration::create(m_fontFaceRule->mutableProperties(), *this);
    return *m_propertiesCSSOMWrapper;
}

String CSSFontFaceRule::cssText() const
{
    return cssTextInternal(m_fontFaceRule->properties().asText());
}

String CSSFontFaceRule::cssTextWithReplacementURLs(const UncheckedKeyHashMap<String, String>& replacementURLStrings, const UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>&) const
{
    auto mutableStyleProperties = m_fontFaceRule->properties().mutableCopy();
    mutableStyleProperties->setReplacementURLForSubresources(replacementURLStrings);
    auto declarations = mutableStyleProperties->asText();
    mutableStyleProperties->clearReplacementURLForSubresources();

    return cssTextInternal(declarations);
}

String CSSFontFaceRule::cssTextInternal(const String& declarations) const
{
    if (declarations.isEmpty())
        return "@font-face { }"_s;

    return makeString("@font-face { "_s, declarations, " }"_s);
}

void CSSFontFaceRule::reattach(StyleRuleBase& rule)
{
    m_fontFaceRule = downcast<StyleRuleFontFace>(rule);
    if (m_propertiesCSSOMWrapper)
        m_propertiesCSSOMWrapper->reattach(m_fontFaceRule->mutableProperties());
}

} // namespace WebCore
