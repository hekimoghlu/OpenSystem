/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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

#include "MatchResult.h"
#include "StyleScopeRuleSets.h"
#include <wtf/Forward.h>

namespace WebCore {

class StyleRulePage;

namespace Style {

class PageRuleCollector {
public:
    PageRuleCollector(ScopeRuleSets& ruleSets, WritingMode rootWritingMode)
        : m_ruleSets(ruleSets)
        , m_rootWritingMode(rootWritingMode)
    { }

    void matchAllPageRules(int pageIndex);
    const MatchResult& matchResult() const { return m_result; }

private:
    bool isLeftPage(int pageIndex) const;
    bool isRightPage(int pageIndex) const { return !isLeftPage(pageIndex); }
    bool isFirstPage(int pageIndex) const;
    String pageName(int pageIndex) const;

    void matchPageRules(RuleSet* rules, bool isLeftPage, bool isFirstPage, const String& pageName);
    void matchPageRulesForList(Vector<StyleRulePage*>& matchedRules, const Vector<StyleRulePage*>& rules, bool isLeftPage, bool isFirstPage, const String& pageName);

    ScopeRuleSets& m_ruleSets;
    WritingMode m_rootWritingMode;

    MatchResult m_result;
};

} // namespace Style
} // namespace WebCore
