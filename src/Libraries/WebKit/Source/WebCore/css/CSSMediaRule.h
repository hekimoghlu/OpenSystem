/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#pragma once

#include "CSSConditionRule.h"

namespace WebCore {

class MediaList;
class StyleRuleMedia;

namespace MQ {
struct MediaQuery;
using MediaQueryList = Vector<MediaQuery>;
}

class CSSMediaRule final : public CSSConditionRule {
public:
    static Ref<CSSMediaRule> create(StyleRuleMedia& rule, CSSStyleSheet* sheet) { return adoptRef(*new CSSMediaRule(rule, sheet)); }
    virtual ~CSSMediaRule();

    WEBCORE_EXPORT MediaList* media() const;

private:
    friend class MediaList;

    CSSMediaRule(StyleRuleMedia&, CSSStyleSheet*);

    StyleRuleType styleRuleType() const final { return StyleRuleType::Media; }
    String cssText() const final;
    String cssTextWithReplacementURLs(const UncheckedKeyHashMap<String, String>&, const UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>&) const final;
    String conditionText() const final;

    const MQ::MediaQueryList& mediaQueries() const;
    void setMediaQueries(MQ::MediaQueryList&&);

    mutable RefPtr<MediaList> m_mediaCSSOMWrapper;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_RULE(CSSMediaRule, StyleRuleType::Media)
