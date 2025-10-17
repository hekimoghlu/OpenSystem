/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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

#include "CSSParserEnum.h"
#include "ExceptionOr.h"
#include "StyleRuleType.h"
#include <wtf/TypeCasts.h>

namespace WebCore {

class CSSStyleSheet;
class StyleRuleBase;
class StyleRule;
class StyleRuleWithNesting;

struct CSSParserContext;

class CSSRule : public RefCounted<CSSRule> {
public:
    virtual ~CSSRule() = default;

    WEBCORE_EXPORT unsigned short typeForCSSOM() const;

    virtual StyleRuleType styleRuleType() const = 0;
    virtual bool isGroupingRule() const { return false; }
    virtual String cssText() const = 0;
    virtual String cssTextWithReplacementURLs(const UncheckedKeyHashMap<String, String>&, const UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String>&) const { return cssText(); }
    virtual void reattach(StyleRuleBase&) = 0;

    void setParentStyleSheet(CSSStyleSheet*);
    void setParentRule(CSSRule*);
    CSSStyleSheet* parentStyleSheet() const;
    CSSRule* parentRule() const { return m_parentIsRule ? m_parentRule : nullptr; }
    bool hasStyleRuleAncestor() const;
    CSSParserEnum::NestedContext nestedContext() const;
    virtual RefPtr<StyleRuleWithNesting> prepareChildStyleRuleForNesting(StyleRule&);
    virtual void getChildStyleSheets(UncheckedKeyHashSet<RefPtr<CSSStyleSheet>>&) { }

    WEBCORE_EXPORT ExceptionOr<void> setCssText(const String&);

protected:
    explicit CSSRule(CSSStyleSheet*);

    bool hasCachedSelectorText() const { return m_hasCachedSelectorText; }
    void setHasCachedSelectorText(bool hasCachedSelectorText) const { m_hasCachedSelectorText = hasCachedSelectorText; }

    const CSSParserContext& parserContext() const;

private:
    mutable unsigned char m_hasCachedSelectorText : 1;
    unsigned char m_parentIsRule : 1;
    union {
        CSSRule* m_parentRule;
        CSSStyleSheet* m_parentStyleSheet;
    };
};

inline CSSRule::CSSRule(CSSStyleSheet* parent)
    : m_hasCachedSelectorText(false)
    , m_parentIsRule(false)
    , m_parentStyleSheet(parent)
{
}

inline void CSSRule::setParentStyleSheet(CSSStyleSheet* styleSheet)
{
    m_parentIsRule = false;
    m_parentStyleSheet = styleSheet;
}

inline void CSSRule::setParentRule(CSSRule* rule)
{
    m_parentIsRule = true;
    m_parentRule = rule;
}

inline CSSStyleSheet* CSSRule::parentStyleSheet() const
{
    if (m_parentIsRule)
        return m_parentRule ? m_parentRule->parentStyleSheet() : nullptr;
    return m_parentStyleSheet;
}

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_CSS_RULE(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToValueTypeName) \
    static bool isType(const WebCore::CSSRule& rule) { return rule.styleRuleType() == WebCore::predicate; } \
SPECIALIZE_TYPE_TRAITS_END()
