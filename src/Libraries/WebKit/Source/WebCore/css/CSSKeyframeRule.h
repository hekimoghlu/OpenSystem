/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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

namespace WebCore {

class CSSStyleDeclaration;
class CSSKeyframesRule;
class StyleProperties;
class StyleRuleCSSStyleDeclaration;

class StyleRuleKeyframe final : public StyleRuleBase {
public:
    static Ref<StyleRuleKeyframe> create(Ref<StyleProperties>&&);
    static Ref<StyleRuleKeyframe> create(Vector<double>&& keys, Ref<StyleProperties>&&);
    ~StyleRuleKeyframe();

    Ref<StyleRuleKeyframe> copy() const { RELEASE_ASSERT_NOT_REACHED(); }

    String keyText() const;
    bool setKeyText(const String&);
    void setKey(double key)
    {
        ASSERT(m_keys.isEmpty());
        m_keys.clear();
        m_keys.append(key);
    }

    const Vector<double>& keys() const { return m_keys; };

    const StyleProperties& properties() const { return m_properties; }
    MutableStyleProperties& mutableProperties();

    String cssText() const;

private:
    explicit StyleRuleKeyframe(Ref<StyleProperties>&&);
    StyleRuleKeyframe(Vector<double>&&, Ref<StyleProperties>&&);

    Ref<StyleProperties> m_properties;
    Vector<double> m_keys;
};

class CSSKeyframeRule final : public CSSRule {
public:
    virtual ~CSSKeyframeRule();

    String cssText() const final { return m_keyframe->cssText(); }
    void reattach(StyleRuleBase&) final;

    String keyText() const { return m_keyframe->keyText(); }
    void setKeyText(const String& text) { m_keyframe->setKeyText(text); }

    CSSStyleDeclaration& style();

private:
    CSSKeyframeRule(StyleRuleKeyframe&, CSSKeyframesRule* parent);

    StyleRuleType styleRuleType() const final { return StyleRuleType::Keyframe; }

    Ref<StyleRuleKeyframe> m_keyframe;
    mutable RefPtr<StyleRuleCSSStyleDeclaration> m_propertiesCSSOMWrapper;
    
    friend class CSSKeyframesRule;
};

} // namespace WebCore
