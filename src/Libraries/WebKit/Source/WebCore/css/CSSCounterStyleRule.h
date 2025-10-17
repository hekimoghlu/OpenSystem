/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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

#include "CSSCounterStyle.h"
#include "CSSRule.h"
#include "StyleProperties.h"
#include "StyleRule.h"
#include <wtf/text/AtomString.h>

namespace WebCore {
class StyleRuleCounterStyle final : public StyleRuleBase {
public:
    static Ref<StyleRuleCounterStyle> create(const AtomString&, CSSCounterStyleDescriptors&&);
    ~StyleRuleCounterStyle();

    Ref<StyleRuleCounterStyle> copy() const { return adoptRef(*new StyleRuleCounterStyle(*this)); }

    const CSSCounterStyleDescriptors& descriptors() const { return m_descriptors; };
    CSSCounterStyleDescriptors& mutableDescriptors() { return m_descriptors; };

    const AtomString& name() const { return m_name; }
    String system() const { return m_descriptors.systemCSSText(); }
    String negative() const { return m_descriptors.negativeCSSText(); }
    String prefix() const { return m_descriptors.prefixCSSText(); }
    String suffix() const { return m_descriptors.suffixCSSText(); }
    String range() const { return { m_descriptors.rangesCSSText() }; }
    String pad() const { return m_descriptors.padCSSText(); }
    String fallback() const { return m_descriptors.fallbackCSSText(); }
    String symbols() const { return m_descriptors.symbolsCSSText(); }
    String additiveSymbols() const { return m_descriptors.additiveSymbolsCSSText(); }
    String speakAs() const { return { }; }
    bool newValueInvalidOrEqual(CSSPropertyID, const RefPtr<CSSValue> newValue) const;

    void setName(const AtomString& name) { m_name = name; }

private:
    explicit StyleRuleCounterStyle(const AtomString&, CSSCounterStyleDescriptors&&);
    StyleRuleCounterStyle(const StyleRuleCounterStyle&) = default;

    AtomString m_name;
    CSSCounterStyleDescriptors m_descriptors;
};

class CSSCounterStyleRule final : public CSSRule {
public:
    static Ref<CSSCounterStyleRule> create(StyleRuleCounterStyle&, CSSStyleSheet*);
    virtual ~CSSCounterStyleRule();

    String cssText() const final;
    void reattach(StyleRuleBase&) final;
    StyleRuleType styleRuleType() const final { return StyleRuleType::CounterStyle; }

    String name() const { return m_counterStyleRule->name(); }
    String system() const { return m_counterStyleRule->system(); }
    String negative() const { return m_counterStyleRule->negative(); }
    String prefix() const { return m_counterStyleRule->prefix(); }
    String suffix() const { return m_counterStyleRule->suffix(); }
    String range() const { return m_counterStyleRule->range(); }
    String pad() const { return m_counterStyleRule->pad(); }
    String fallback() const { return m_counterStyleRule->fallback(); }
    String symbols() const { return m_counterStyleRule->symbols(); }
    String additiveSymbols() const { return m_counterStyleRule->additiveSymbols(); }
    String speakAs() const { return m_counterStyleRule->speakAs(); }

    void setName(const String&);
    void setSystem(const String&);
    void setNegative(const String&);
    void setPrefix(const String&);
    void setSuffix(const String&);
    void setRange(const String&);
    void setPad(const String&);
    void setFallback(const String&);
    void setSymbols(const String&);
    void setAdditiveSymbols(const String&);
    void setSpeakAs(const String&);

private:
    CSSCounterStyleRule(StyleRuleCounterStyle&, CSSStyleSheet* parent);

    bool setterInternal(CSSPropertyID, const String&);
    RefPtr<CSSValue> cssValueFromText(CSSPropertyID, const String&);
    const CSSCounterStyleDescriptors& descriptors() const { return m_counterStyleRule->descriptors(); }
    CSSCounterStyleDescriptors& mutableDescriptors() { return m_counterStyleRule->mutableDescriptors(); }

    Ref<StyleRuleCounterStyle> m_counterStyleRule;
};

CSSCounterStyleDescriptors::System toCounterStyleSystemEnum(const CSSValue*);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_RULE(CSSCounterStyleRule, StyleRuleType::CounterStyle)

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::StyleRuleCounterStyle)
static bool isType(const WebCore::StyleRuleBase& rule) { return rule.isCounterStyleRule(); }
SPECIALIZE_TYPE_TRAITS_END()
