/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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
#include "CSSPropertyRule.h"

#include "CSSMarkup.h"
#include "CSSParserTokenRange.h"
#include "CSSStyleSheet.h"
#include "StyleRule.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSPropertyRule::CSSPropertyRule(StyleRuleProperty& rule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_propertyRule(rule)
{
}

Ref<CSSPropertyRule> CSSPropertyRule::create(StyleRuleProperty& rule, CSSStyleSheet* parent)
{
    return adoptRef(*new CSSPropertyRule(rule, parent));
}

CSSPropertyRule::~CSSPropertyRule() = default;

String CSSPropertyRule::name() const
{
    return m_propertyRule->descriptor().name;
}

String CSSPropertyRule::syntax() const
{
    return m_propertyRule->descriptor().syntax;
}

bool CSSPropertyRule::inherits() const
{
    return m_propertyRule->descriptor().inherits.value_or(false);
}

String CSSPropertyRule::initialValue() const
{
    if (!m_propertyRule->descriptor().initialValue)
        return nullString();

    return m_propertyRule->descriptor().initialValue->serialize();
}

String CSSPropertyRule::cssText() const
{
    StringBuilder builder;

    auto& descriptor = m_propertyRule->descriptor();

    builder.append("@property "_s);
    serializeIdentifier(descriptor.name, builder);
    builder.append(" { "_s);

    if (!descriptor.syntax.isNull()) {
        builder.append("syntax: "_s);
        serializeString(syntax(), builder);
        builder.append("; "_s);
    }

    if (descriptor.inherits)
        builder.append("inherits: "_s, *descriptor.inherits ? "true"_s : "false"_s, "; "_s);

    if (descriptor.initialValue)
        builder.append("initial-value: "_s, initialValue(), "; "_s);

    builder.append('}');

    return builder.toString();
}

void CSSPropertyRule::reattach(StyleRuleBase& rule)
{
    m_propertyRule = downcast<StyleRuleProperty>(rule);
}

} // namespace WebCore
