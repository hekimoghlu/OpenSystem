/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
#include "CSSMediaRule.h"

#include "CSSStyleSheet.h"
#include "MediaList.h"
#include "MediaQueryParser.h"
#include "StyleRule.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSMediaRule::CSSMediaRule(StyleRuleMedia& mediaRule, CSSStyleSheet* parent)
    : CSSConditionRule(mediaRule, parent)
{
}

CSSMediaRule::~CSSMediaRule()
{
    if (m_mediaCSSOMWrapper)
        m_mediaCSSOMWrapper->detachFromParent();
}

const MQ::MediaQueryList& CSSMediaRule::mediaQueries() const
{
    return downcast<StyleRuleMedia>(groupRule()).mediaQueries();
}

void CSSMediaRule::setMediaQueries(MQ::MediaQueryList&& queries)
{
    downcast<StyleRuleMedia>(groupRule()).setMediaQueries(WTFMove(queries));
}

String CSSMediaRule::cssText() const
{
    StringBuilder builder;
    builder.append("@media "_s, conditionText());
    appendCSSTextForItems(builder);
    return builder.toString();
}

String CSSMediaRule::cssTextWithReplacementURLs(const UncheckedKeyHashMap<String, String>& replacementURLStrings, const UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>& replacementURLStringsForCSSStyleSheet) const
{
    StringBuilder builder;
    builder.append("@media "_s, conditionText());
    appendCSSTextWithReplacementURLsForItems(builder, replacementURLStrings, replacementURLStringsForCSSStyleSheet);
    return builder.toString();
}

String CSSMediaRule::conditionText() const
{
    StringBuilder builder;
    MQ::serialize(builder, mediaQueries());
    return builder.toString();
}

MediaList* CSSMediaRule::media() const
{
    if (!m_mediaCSSOMWrapper)
        m_mediaCSSOMWrapper = MediaList::create(const_cast<CSSMediaRule*>(this));
    return m_mediaCSSOMWrapper.get();
}

} // namespace WebCore
