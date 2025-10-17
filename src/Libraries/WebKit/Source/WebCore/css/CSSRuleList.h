/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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

#include <wtf/AbstractRefCounted.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>

namespace WebCore {

class CSSRule;
class CSSStyleSheet;

class CSSRuleList : public AbstractRefCounted {
    WTF_MAKE_NONCOPYABLE(CSSRuleList);
public:
    virtual ~CSSRuleList();

    virtual unsigned length() const = 0;
    virtual CSSRule* item(unsigned index) const = 0;
    bool isSupportedPropertyIndex(unsigned index) const { return item(index); }
    
    virtual CSSStyleSheet* styleSheet() const = 0;
    
protected:
    CSSRuleList();
};

class StaticCSSRuleList final : public CSSRuleList, public RefCounted<StaticCSSRuleList> {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<StaticCSSRuleList> create() { return adoptRef(*new StaticCSSRuleList); }

    Vector<RefPtr<CSSRule>>& rules() { return m_rules; }
    
    CSSStyleSheet* styleSheet() const final { return nullptr; }

    ~StaticCSSRuleList();
private:    
    StaticCSSRuleList();

    unsigned length() const final { return m_rules.size(); }
    CSSRule* item(unsigned index) const final { return index < m_rules.size() ? m_rules[index].get() : nullptr; }

    Vector<RefPtr<CSSRule>> m_rules;
};

// The rule owns the live list.
template <class Rule>
class LiveCSSRuleList final : public CSSRuleList {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(LiveCSSRuleList);
public:
    LiveCSSRuleList(Rule& rule)
        : m_rule(rule)
    {
    }
    
    void ref() const final { m_rule.ref(); }
    void deref() const final { m_rule.deref(); }

private:
    unsigned length() const final { return m_rule.length(); }
    CSSRule* item(unsigned index) const final { return m_rule.item(index); }
    CSSStyleSheet* styleSheet() const final { return m_rule.parentStyleSheet(); }
    
    Rule& m_rule;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<class Rule>, LiveCSSRuleList<Rule>);

} // namespace WebCore
