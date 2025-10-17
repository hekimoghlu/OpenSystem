/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 18, 2025.
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
#include "CSSLayerBlockRule.h"

#include "CSSMarkup.h"
#include "CSSStyleSheet.h"
#include "StyleRule.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSLayerBlockRule::CSSLayerBlockRule(StyleRuleLayer& rule, CSSStyleSheet* parent)
    : CSSGroupingRule(rule, parent)
{
    ASSERT(!rule.isStatement());
}

Ref<CSSLayerBlockRule> CSSLayerBlockRule::create(StyleRuleLayer& rule, CSSStyleSheet* parent)
{
    return adoptRef(*new CSSLayerBlockRule(rule, parent));
}

String CSSLayerBlockRule::cssText() const
{
    StringBuilder builder;

    builder.append("@layer"_s);
    if (auto name = this->name(); !name.isEmpty())
        builder.append(' ', name);
    appendCSSTextForItems(builder);
    return builder.toString();
}

String CSSLayerBlockRule::name() const
{
    auto& layer = downcast<StyleRuleLayer>(groupRule());

    if (layer.name().isEmpty())
        return emptyString();

    return stringFromCascadeLayerName(layer.name());
}

String stringFromCascadeLayerName(const CascadeLayerName& name)
{
    StringBuilder result;
    for (auto& segment : name) {
        serializeIdentifier(segment, result);
        if (&segment != &name.last())
            result.append('.');
    }
    return result.toString();
}

} // namespace WebCore

