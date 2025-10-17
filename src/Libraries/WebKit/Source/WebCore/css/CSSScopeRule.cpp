/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
#include "CSSScopeRule.h"

#include "StyleRule.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSScopeRule::CSSScopeRule(StyleRuleScope& rule, CSSStyleSheet* parent)
    : CSSGroupingRule(rule, parent)
{
}

Ref<CSSScopeRule> CSSScopeRule::create(StyleRuleScope& rule, CSSStyleSheet* parent)
{
    return adoptRef(*new CSSScopeRule(rule, parent));
}

const StyleRuleScope& CSSScopeRule::styleRuleScope() const
{
    return downcast<StyleRuleScope>(groupRule());
}

String CSSScopeRule::cssText() const
{
    StringBuilder builder;
    builder.append("@scope"_s);
    auto start = this->start();
    if (!start.isEmpty())
        builder.append(" ("_s, start, ')');
    auto end = this->end();
    if (!end.isEmpty())
        builder.append(" to "_s, '(', end, ')');
    appendCSSTextForItems(builder);
    return builder.toString();
}

String CSSScopeRule::start() const
{
    auto& scope = styleRuleScope().originalScopeStart();
    if (scope.isEmpty())
        return { };

    return scope.selectorsText();
}

String CSSScopeRule::end() const
{
    auto& scope = styleRuleScope().originalScopeEnd();
    if (scope.isEmpty())
        return { };

    return scope.selectorsText();
}

} // namespace WebCore
