/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#include "CSSLayerStatementRule.h"

#include "CSSLayerBlockRule.h"
#include "CSSStyleSheet.h"
#include "StyleRule.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSLayerStatementRule::CSSLayerStatementRule(StyleRuleLayer& rule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_layerRule(rule)
{
    ASSERT(rule.isStatement());
}

Ref<CSSLayerStatementRule> CSSLayerStatementRule::create(StyleRuleLayer& rule, CSSStyleSheet* parent)
{
    return adoptRef(*new CSSLayerStatementRule(rule, parent));
}

CSSLayerStatementRule::~CSSLayerStatementRule() = default;

String CSSLayerStatementRule::cssText() const
{
    StringBuilder result;

    result.append("@layer "_s);

    auto nameList = this->nameList();
    for (auto& name : nameList) {
        result.append(name);
        if (&name != &nameList.last())
            result.append(", "_s);
    }
    result.append(';');

    return result.toString();
}

Vector<String> CSSLayerStatementRule::nameList() const
{
    Vector<String> result;

    for (auto& name : m_layerRule.get().nameList())
        result.append(stringFromCascadeLayerName(name));

    return result;
}

void CSSLayerStatementRule::reattach(StyleRuleBase& rule)
{
    m_layerRule = downcast<StyleRuleLayer>(rule);
}

} // namespace WebCore

