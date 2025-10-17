/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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

#include "CSSRule.h"

namespace WebCore {

class MediaList;
class StyleRuleImport;

namespace MQ {
struct MediaQuery;
using MediaQueryList = Vector<MediaQuery>;
}

class CSSImportRule final : public CSSRule, public CanMakeWeakPtr<CSSImportRule> {
public:
    static Ref<CSSImportRule> create(StyleRuleImport& rule, CSSStyleSheet* sheet) { return adoptRef(*new CSSImportRule(rule, sheet)); }

    virtual ~CSSImportRule();

    WEBCORE_EXPORT String href() const;
    WEBCORE_EXPORT MediaList& media() const;
    WEBCORE_EXPORT CSSStyleSheet* styleSheet() const;
    RefPtr<CSSStyleSheet> protectedStyleSheet() const;
    String layerName() const;
    String supportsText() const;

private:
    friend class MediaList;

    CSSImportRule(StyleRuleImport&, CSSStyleSheet*);

    StyleRuleType styleRuleType() const final { return StyleRuleType::Import; }
    String cssText() const final;
    String cssTextWithReplacementURLs(const UncheckedKeyHashMap<String, String>&, const UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>&) const final;
    void reattach(StyleRuleBase&) final;
    void getChildStyleSheets(UncheckedKeyHashSet<RefPtr<CSSStyleSheet>>&) final;

    String cssTextInternal(const String& urlString) const;
    const MQ::MediaQueryList& mediaQueries() const;
    void setMediaQueries(MQ::MediaQueryList&&);

    Ref<StyleRuleImport> m_importRule;
    mutable RefPtr<MediaList> m_mediaCSSOMWrapper;
    mutable RefPtr<CSSStyleSheet> m_styleSheetCSSOMWrapper;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_RULE(CSSImportRule, StyleRuleType::Import)
