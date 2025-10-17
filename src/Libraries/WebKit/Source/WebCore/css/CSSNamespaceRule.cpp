/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 19, 2024.
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
#include "CSSNamespaceRule.h"

#include "CSSMarkup.h"
#include "StyleRule.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSNamespaceRule::CSSNamespaceRule(StyleRuleNamespace& namespaceRule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_namespaceRule(namespaceRule)
{
}

CSSNamespaceRule::~CSSNamespaceRule() = default;

AtomString CSSNamespaceRule::namespaceURI() const
{
    return m_namespaceRule->uri();
}
    
AtomString CSSNamespaceRule::prefix() const
{
    return m_namespaceRule->prefix();
}

String CSSNamespaceRule::cssText() const
{
    auto prefix = this->prefix();
    StringBuilder result;
    result.append("@namespace "_s);
    serializeIdentifier(prefix, result);
    result.append(prefix.isEmpty() ? ""_s : " "_s, "url("_s, serializeString(namespaceURI()), ");"_s);
    return result.toString();
}

void CSSNamespaceRule::reattach(StyleRuleBase&)
{
}

} // namespace WebCore
