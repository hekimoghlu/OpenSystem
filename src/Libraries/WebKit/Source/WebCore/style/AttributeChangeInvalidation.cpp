/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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
#include "AttributeChangeInvalidation.h"

#include "ElementIterator.h"
#include "ElementRareData.h"
#include "StyleInvalidationFunctions.h"

namespace WebCore {
namespace Style {

static bool mayBeAffectedByAttributeChange(const RuleFeatureSet& features, bool isHTML, const QualifiedName& attributeName)
{
    auto& nameSet = isHTML ? features.attributeLowercaseLocalNamesInRules : features.attributeLocalNamesInRules;
    return nameSet.contains(attributeName.localName());
}

void AttributeChangeInvalidation::invalidateStyle(const QualifiedName& attributeName, const AtomString& oldValue, const AtomString& newValue)
{
    if (newValue == oldValue)
        return;

    bool isHTML = m_element.isHTMLElement() && m_element.document().isHTMLDocument();

    bool shouldInvalidateCurrent = false;
    bool mayAffectStyleInShadowTree = false;

    auto attributeNameForLookups = attributeName.localNameLowercase();

    traverseRuleFeatures(m_element, [&] (const RuleFeatureSet& features, bool mayAffectShadowTree) {
        if (mayAffectShadowTree && mayBeAffectedByAttributeChange(features, isHTML, attributeName))
            mayAffectStyleInShadowTree = true;
        if (features.attributesAffectingHost.contains(attributeNameForLookups))
            shouldInvalidateCurrent = true;
        else if (features.contentAttributeNamesInRules.contains(attributeNameForLookups))
            shouldInvalidateCurrent = true;
    });

    if (mayAffectStyleInShadowTree) {
        // FIXME: More fine-grained invalidation.
        m_element.invalidateStyleForSubtree();
    }

    if (shouldInvalidateCurrent)
        m_element.invalidateStyle();

    auto collect = [&](auto& ruleSets, std::optional<MatchElement> onlyMatchElement = { }) {
        auto* invalidationRuleSets = ruleSets.attributeInvalidationRuleSets(attributeNameForLookups);
        if (!invalidationRuleSets)
            return;

        for (auto& invalidationRuleSet : *invalidationRuleSets) {
            if (onlyMatchElement && invalidationRuleSet.matchElement != onlyMatchElement)
                continue;

            for (auto* selector : invalidationRuleSet.invalidationSelectors) {
                if (!selector->isAttributeSelector()) {
                    ASSERT_NOT_REACHED();
                    continue;
                }
                bool oldMatches = !oldValue.isNull() && SelectorChecker::attributeSelectorMatches(m_element, attributeName, oldValue, *selector);
                bool newMatches = !newValue.isNull() && SelectorChecker::attributeSelectorMatches(m_element, attributeName, newValue, *selector);
                if (oldMatches != newMatches) {
                    Invalidator::addToMatchElementRuleSets(m_matchElementRuleSets, invalidationRuleSet);
                    break;
                }
            }
        }
    };

    collect(m_element.styleResolver().ruleSets());

    if (auto* shadowRoot = m_element.shadowRoot())
        collect(shadowRoot->styleScope().resolver().ruleSets(), MatchElement::Host);
}

void AttributeChangeInvalidation::invalidateStyleWithRuleSets()
{
    Invalidator::invalidateWithMatchElementRuleSets(m_element, m_matchElementRuleSets);
}


}
}
