/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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
#include "PageRuleCollector.h"

#include "CommonAtomStrings.h"
#include "StyleProperties.h"
#include "StyleRule.h"
#include "UserAgentStyle.h"

namespace WebCore {
namespace Style {

static inline bool comparePageRules(const StyleRulePage* r1, const StyleRulePage* r2)
{
    return r1->selector()->specificityForPage() < r2->selector()->specificityForPage();
}

bool PageRuleCollector::isLeftPage(int pageIndex) const
{
    bool isFirstPageLeft = !m_rootWritingMode.isAnyLeftToRight();
    return (pageIndex + (isFirstPageLeft ? 1 : 0)) % 2;
}

bool PageRuleCollector::isFirstPage(int pageIndex) const
{
    // FIXME: In case of forced left/right page, page at index 1 (not 0) can be the first page.
    return (!pageIndex);
}

String PageRuleCollector::pageName(int /* pageIndex */) const
{
    // FIXME: Implement page index to page name mapping.
    return emptyString();
}

void PageRuleCollector::matchAllPageRules(int pageIndex)
{
    const bool isLeft = isLeftPage(pageIndex);
    const bool isFirst = isFirstPage(pageIndex);
    const String page = pageName(pageIndex);
    
    matchPageRules(UserAgentStyle::defaultPrintStyle, isLeft, isFirst, page);
    matchPageRules(m_ruleSets.userStyle(), isLeft, isFirst, page);
    // Only consider the global author RuleSet for @page rules, as per the HTML5 spec.
    if (m_ruleSets.isAuthorStyleDefined())
        matchPageRules(&m_ruleSets.authorStyle(), isLeft, isFirst, page);
}

void PageRuleCollector::matchPageRules(RuleSet* rules, bool isLeftPage, bool isFirstPage, const String& pageName)
{
    if (!rules)
        return;

    Vector<StyleRulePage*> matchedPageRules;
    matchPageRulesForList(matchedPageRules, rules->pageRules(), isLeftPage, isFirstPage, pageName);
    if (matchedPageRules.isEmpty())
        return;

    std::stable_sort(matchedPageRules.begin(), matchedPageRules.end(), comparePageRules);

    m_result.authorDeclarations.appendContainerWithMapping(matchedPageRules, [](auto& pageRule) {
        return MatchedProperties { pageRule->properties() };
    });
}

static bool checkPageSelectorComponents(const CSSSelector* selector, bool isLeftPage, bool isFirstPage, const String& pageName)
{
    for (const CSSSelector* component = selector; component; component = component->tagHistory()) {
        if (component->match() == CSSSelector::Match::Tag) {
            const AtomString& localName = component->tagQName().localName();
            if (localName != starAtom() && localName != pageName)
                return false;
        } else if (component->match() == CSSSelector::Match::PagePseudoClass) {
            auto pseudoType = component->pagePseudoClass();
            if ((pseudoType == CSSSelector::PagePseudoClass::Left && !isLeftPage)
                || (pseudoType == CSSSelector::PagePseudoClass::Right && isLeftPage)
                || (pseudoType == CSSSelector::PagePseudoClass::First && !isFirstPage))
            {
                return false;
            }
        }
    }
    return true;
}

void PageRuleCollector::matchPageRulesForList(Vector<StyleRulePage*>& matchedRules, const Vector<StyleRulePage*>& rules, bool isLeftPage, bool isFirstPage, const String& pageName)
{
    for (unsigned i = 0; i < rules.size(); ++i) {
        StyleRulePage* rule = rules[i];

        if (!checkPageSelectorComponents(rule->selector(), isLeftPage, isFirstPage, pageName))
            continue;

        // If the rule has no properties to apply, then ignore it.
        const StyleProperties& properties = rule->properties();
        if (properties.isEmpty())
            continue;

        // Add this rule to our list of matched rules.
        matchedRules.append(rule);
    }
}

} // namespace Style
} // namespace WebCore
