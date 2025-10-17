/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
#include "IdChangeInvalidation.h"

#include "ElementChildIteratorInlines.h"
#include "ElementRareData.h"
#include "StyleInvalidationFunctions.h"

namespace WebCore {
namespace Style {

void IdChangeInvalidation::invalidateStyle(const AtomString& changedId)
{
    if (changedId.isEmpty())
        return;

    bool mayAffectStyle = false;
    bool mayAffectStyleInShadowTree = false;

    traverseRuleFeatures(m_element, [&] (const RuleFeatureSet& features, bool mayAffectShadowTree) {
        if (!features.idsInRules.contains(changedId))
            return;
        mayAffectStyle = true;
        if (mayAffectShadowTree)
            mayAffectStyleInShadowTree = true;
    });

    if (!mayAffectStyle)
        return;

    if (mayAffectStyleInShadowTree) {
        m_element.invalidateStyleForSubtree();
        return;
    }

    m_element.invalidateStyle();

    auto collect = [&](auto& ruleSets, std::optional<MatchElement> onlyMatchElement = { }) {
        // This could be easily optimized for fine-grained descendant invalidation similar to ClassChangeInvalidation.
        // However using ids for dynamic styling is rare and this is probably not worth the memory cost of the required data structures.
        bool mayAffectDescendantStyle = ruleSets.features().idsMatchingAncestorsInRules.contains(changedId);
        if (mayAffectDescendantStyle)
            m_element.invalidateStyleForSubtree();
        else
            m_element.invalidateStyle();

        // Invalidation rulesets exist for :has() / :nth-child() / :nth-last-child.
        if (auto* invalidationRuleSets = ruleSets.idInvalidationRuleSets(changedId)) {
            for (auto& invalidationRuleSet : *invalidationRuleSets) {
                if (onlyMatchElement && invalidationRuleSet.matchElement != onlyMatchElement)
                    continue;

                Invalidator::addToMatchElementRuleSets(m_matchElementRuleSets, invalidationRuleSet);
            }
        }
    };

    collect(m_element.styleResolver().ruleSets());

    if (auto* shadowRoot = m_element.shadowRoot())
        collect(shadowRoot->styleScope().resolver().ruleSets(), MatchElement::Host);

}

void IdChangeInvalidation::invalidateStyleWithRuleSets()
{
    Invalidator::invalidateWithMatchElementRuleSets(m_element, m_matchElementRuleSets);
}

}
}
