/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
#include "CSSRule.h"

#include "CSSScopeRule.h"
#include "CSSStyleRule.h"
#include "CSSStyleSheet.h"
#include "StyleRule.h"
#include "StyleSheetContents.h"
#include "css/parser/CSSParserEnum.h"

namespace WebCore {

struct SameSizeAsCSSRule : public RefCounted<SameSizeAsCSSRule> {
    virtual ~SameSizeAsCSSRule();
    unsigned char bitfields;
    void* pointerUnion;
};

static_assert(sizeof(CSSRule) == sizeof(SameSizeAsCSSRule), "CSSRule should stay small");

unsigned short CSSRule::typeForCSSOM() const
{
    // "This enumeration is thus frozen in its current state, and no new new values will be
    // added to reflect additional at-rules; all at-rules beyond the ones listed above will return 0."
    // https://drafts.csswg.org/cssom/#the-cssrule-interface
    if (styleRuleType() >= firstUnexposedStyleRuleType)
        return 0;

    return enumToUnderlyingType(styleRuleType());
}

ExceptionOr<void> CSSRule::setCssText(const String&)
{
    return { };
}

const CSSParserContext& CSSRule::parserContext() const
{
    RefPtr styleSheet = parentStyleSheet();
    return styleSheet ? styleSheet->contents().parserContext() : strictCSSParserContext();
}

bool CSSRule::hasStyleRuleAncestor() const
{
    auto current = this->parentRule();
    while (current) {
        if (current->styleRuleType() == StyleRuleType::Style)
            return true;

        current = current->parentRule();
    }
    return false;
}

CSSParserEnum::NestedContext CSSRule::nestedContext() const
{
    for (RefPtr parentRule = this->parentRule(); parentRule; parentRule = parentRule->parentRule()) {
        if (is<CSSStyleRule>(*parentRule))
            return CSSParserEnum::NestedContextType::Style;
        if (is<CSSScopeRule>(*parentRule))
            return CSSParserEnum::NestedContextType::Scope;
    }

    return { };
}

RefPtr<StyleRuleWithNesting> CSSRule::prepareChildStyleRuleForNesting(StyleRule&)
{
    return nullptr;
}

} // namespace WebCore
