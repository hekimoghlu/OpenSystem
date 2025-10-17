/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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
#include "StyleRule.h"
#include <memory>
#include <wtf/Forward.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class CSSKeyframeRule;
class CSSRuleList;
class StyleRuleKeyframe;

class StyleRuleKeyframes final : public StyleRuleBase {
public:
    static Ref<StyleRuleKeyframes> create(const AtomString& name);
    ~StyleRuleKeyframes();
    
    const Vector<Ref<StyleRuleKeyframe>>& keyframes() const;

    void parserAppendKeyframe(RefPtr<StyleRuleKeyframe>&&);
    void wrapperAppendKeyframe(Ref<StyleRuleKeyframe>&&);
    void wrapperRemoveKeyframe(unsigned);

    const AtomString& name() const { return m_name; }
    void setName(const AtomString& name) { m_name = name; }

    std::optional<size_t> findKeyframeIndex(const String& key) const;

    Ref<StyleRuleKeyframes> copy() const { return adoptRef(*new StyleRuleKeyframes(*this)); }

    void shrinkToFit();

private:
    explicit StyleRuleKeyframes(const AtomString&);
    StyleRuleKeyframes(const StyleRuleKeyframes&);
    
    mutable Vector<Ref<StyleRuleKeyframe>> m_keyframes;
    AtomString m_name;
};

class CSSKeyframesRule final : public CSSRule {
public:
    static Ref<CSSKeyframesRule> create(StyleRuleKeyframes& rule, CSSStyleSheet* sheet) { return adoptRef(*new CSSKeyframesRule(rule, sheet)); }

    virtual ~CSSKeyframesRule();

    StyleRuleType styleRuleType() const final { return StyleRuleType::Keyframes; }
    String cssText() const final;
    void reattach(StyleRuleBase&) final;

    const AtomString& name() const { return m_keyframesRule->name(); }
    void setName(const AtomString&);

    CSSRuleList& cssRules();

    void appendRule(const String& rule);
    void deleteRule(const String& key);
    CSSKeyframeRule* findRule(const String& key);

    // For IndexedGetter and CSSRuleList.
    unsigned length() const;
    CSSKeyframeRule* item(unsigned index) const;
    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }

private:
    CSSKeyframesRule(StyleRuleKeyframes&, CSSStyleSheet* parent);

    Ref<StyleRuleKeyframes> m_keyframesRule;
    mutable Vector<RefPtr<CSSKeyframeRule>> m_childRuleCSSOMWrappers;
    mutable std::unique_ptr<CSSRuleList> m_ruleListCSSOMWrapper;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_RULE(CSSKeyframesRule, StyleRuleType::Keyframes)

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::StyleRuleKeyframes)
    static bool isType(const WebCore::StyleRuleBase& rule) { return rule.isKeyframesRule(); }
SPECIALIZE_TYPE_TRAITS_END()
