/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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
#include "CSSKeyframeRule.h"

#include "CSSKeyframesRule.h"
#include "CSSParser.h"
#include "MutableStyleProperties.h"
#include "PropertySetCSSStyleDeclaration.h"
#include "StyleProperties.h"
#include "StylePropertiesInlines.h"
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

StyleRuleKeyframe::StyleRuleKeyframe(Ref<StyleProperties>&& properties)
    : StyleRuleBase(StyleRuleType::Keyframe)
    , m_properties(WTFMove(properties))
{
}

StyleRuleKeyframe::StyleRuleKeyframe(Vector<double>&& keys, Ref<StyleProperties>&& properties)
    : StyleRuleBase(StyleRuleType::Keyframe)
    , m_properties(WTFMove(properties))
    , m_keys(WTFMove(keys))
{
}

Ref<StyleRuleKeyframe> StyleRuleKeyframe::create(Ref<StyleProperties>&& properties)
{
    return adoptRef(*new StyleRuleKeyframe(WTFMove(properties)));
}

Ref<StyleRuleKeyframe> StyleRuleKeyframe::create(Vector<double>&& keys, Ref<StyleProperties>&& properties)
{
    return adoptRef(*new StyleRuleKeyframe(WTFMove(keys), WTFMove(properties)));
}

StyleRuleKeyframe::~StyleRuleKeyframe() = default;

MutableStyleProperties& StyleRuleKeyframe::mutableProperties()
{
    if (auto* mutableProperties = dynamicDowncast<MutableStyleProperties>(m_properties.get()))
        return *mutableProperties;
    Ref mutableProperties = m_properties->mutableCopy();
    auto& mutablePropertiesRef = mutableProperties.get();
    m_properties = WTFMove(mutableProperties);
    return mutablePropertiesRef;
}

String StyleRuleKeyframe::keyText() const
{
    StringBuilder keyText;
    for (size_t i = 0; i < m_keys.size(); ++i) {
        if (i)
            keyText.append(',');
        keyText.append(m_keys[i] * 100, '%');
    }
    return keyText.toString();
}
    
bool StyleRuleKeyframe::setKeyText(const String& keyText)
{
    ASSERT(!keyText.isNull());
    auto keys = CSSParser::parseKeyframeKeyList(keyText);
    if (keys.isEmpty())
        return false;
    m_keys = WTFMove(keys);
    return true;
}

String StyleRuleKeyframe::cssText() const
{
    if (auto declarations = m_properties->asText(); !declarations.isEmpty())
        return makeString(keyText(), " { "_s, declarations, " }"_s);
    return makeString(keyText(), " { }"_s);
}

CSSKeyframeRule::CSSKeyframeRule(StyleRuleKeyframe& keyframe, CSSKeyframesRule* parent)
    : CSSRule(nullptr)
    , m_keyframe(keyframe)
{
    setParentRule(parent);
}

CSSKeyframeRule::~CSSKeyframeRule()
{
    if (m_propertiesCSSOMWrapper)
        m_propertiesCSSOMWrapper->clearParentRule();
}

CSSStyleDeclaration& CSSKeyframeRule::style()
{
    if (!m_propertiesCSSOMWrapper)
        m_propertiesCSSOMWrapper = StyleRuleCSSStyleDeclaration::create(m_keyframe->mutableProperties(), *this);
    return *m_propertiesCSSOMWrapper;
}

void CSSKeyframeRule::reattach(StyleRuleBase&)
{
    // No need to reattach, the underlying data is shareable on mutation.
    ASSERT_NOT_REACHED();
}

} // namespace WebCore
