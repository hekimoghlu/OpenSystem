/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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
#include "CSSKeyframesRule.h"

#include "CSSKeyframeRule.h"
#include "CSSParser.h"
#include "CSSRuleList.h"
#include "CSSStyleSheet.h"
#include "Document.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

StyleRuleKeyframes::StyleRuleKeyframes(const AtomString& name)
    : StyleRuleBase(StyleRuleType::Keyframes)
    , m_name(name)
{
}

StyleRuleKeyframes::StyleRuleKeyframes(const StyleRuleKeyframes& o)
    : StyleRuleBase(o)
    , m_keyframes(o.keyframes())
    , m_name(o.m_name)
{
}

Ref<StyleRuleKeyframes> StyleRuleKeyframes::create(const AtomString& name)
{
    return adoptRef(*new StyleRuleKeyframes(name));
}

StyleRuleKeyframes::~StyleRuleKeyframes() = default;

const Vector<Ref<StyleRuleKeyframe>>& StyleRuleKeyframes::keyframes() const
{
    return m_keyframes;
}

void StyleRuleKeyframes::parserAppendKeyframe(RefPtr<StyleRuleKeyframe>&& keyframe)
{
    if (!keyframe)
        return;
    m_keyframes.append(keyframe.releaseNonNull());
}

void StyleRuleKeyframes::wrapperAppendKeyframe(Ref<StyleRuleKeyframe>&& keyframe)
{
    m_keyframes.append(WTFMove(keyframe));
}

void StyleRuleKeyframes::wrapperRemoveKeyframe(unsigned index)
{
    m_keyframes.remove(index);
}

std::optional<size_t> StyleRuleKeyframes::findKeyframeIndex(const String& key) const
{
    auto keys = CSSParser::parseKeyframeKeyList(key);
    if (keys.isEmpty())
        return std::nullopt;
    for (auto i = m_keyframes.size(); i--; ) {
        if (m_keyframes[i]->keys() == keys)
            return i;
    }
    return std::nullopt;
}

void StyleRuleKeyframes::shrinkToFit()
{
    m_keyframes.shrinkToFit();
}

CSSKeyframesRule::CSSKeyframesRule(StyleRuleKeyframes& keyframesRule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_keyframesRule(keyframesRule)
    , m_childRuleCSSOMWrappers(keyframesRule.keyframes().size())
{
}

CSSKeyframesRule::~CSSKeyframesRule()
{
    ASSERT(m_childRuleCSSOMWrappers.size() == m_keyframesRule->keyframes().size());

    for (unsigned i = 0; i < m_childRuleCSSOMWrappers.size(); ++i) {
        if (m_childRuleCSSOMWrappers[i])
            m_childRuleCSSOMWrappers[i]->setParentRule(0);
    }
}

void CSSKeyframesRule::setName(const AtomString& name)
{
    CSSStyleSheet::RuleMutationScope mutationScope(this);

    m_keyframesRule->setName(name);
}

void CSSKeyframesRule::appendRule(const String& ruleText)
{
    ASSERT(m_childRuleCSSOMWrappers.size() == m_keyframesRule->keyframes().size());

    CSSParser parser(parserContext());
    RefPtr<StyleRuleKeyframe> keyframe = parser.parseKeyframeRule(ruleText);
    if (!keyframe)
        return;

    CSSStyleSheet::RuleMutationScope mutationScope(this);

    m_keyframesRule->wrapperAppendKeyframe(keyframe.releaseNonNull());

    m_childRuleCSSOMWrappers.grow(length());
}

void CSSKeyframesRule::deleteRule(const String& s)
{
    ASSERT(m_childRuleCSSOMWrappers.size() == m_keyframesRule->keyframes().size());

    auto i = m_keyframesRule->findKeyframeIndex(s);
    if (!i)
        return;

    CSSStyleSheet::RuleMutationScope mutationScope(this);

    m_keyframesRule->wrapperRemoveKeyframe(*i);

    if (m_childRuleCSSOMWrappers[*i])
        m_childRuleCSSOMWrappers[*i]->setParentRule(nullptr);
    m_childRuleCSSOMWrappers.remove(*i);
}

CSSKeyframeRule* CSSKeyframesRule::findRule(const String& s)
{
    auto i = m_keyframesRule->findKeyframeIndex(s);
    return i ? item(*i) : nullptr;
}

String CSSKeyframesRule::cssText() const
{
    StringBuilder result;
    result.append("@keyframes "_s, name(), " { \n"_s);
    for (unsigned i = 0, size = length(); i < size; ++i)
        result.append("  "_s, m_keyframesRule->keyframes()[i]->cssText(), '\n');
    result.append('}');
    return result.toString();
}

unsigned CSSKeyframesRule::length() const
{ 
    return m_keyframesRule->keyframes().size(); 
}

CSSKeyframeRule* CSSKeyframesRule::item(unsigned index) const
{ 
    if (index >= length())
        return nullptr;
    ASSERT(m_childRuleCSSOMWrappers.size() == m_keyframesRule->keyframes().size());
    auto& rule = m_childRuleCSSOMWrappers[index];
    if (!rule)
        rule = adoptRef(*new CSSKeyframeRule(m_keyframesRule->keyframes()[index], const_cast<CSSKeyframesRule*>(this)));
    return rule.get(); 
}

CSSRuleList& CSSKeyframesRule::cssRules()
{
    if (!m_ruleListCSSOMWrapper)
        m_ruleListCSSOMWrapper = makeUniqueWithoutRefCountedCheck<LiveCSSRuleList<CSSKeyframesRule>>(*this);
    return *m_ruleListCSSOMWrapper;
}

void CSSKeyframesRule::reattach(StyleRuleBase& rule)
{
    m_keyframesRule = downcast<StyleRuleKeyframes>(rule);
}

} // namespace WebCore
