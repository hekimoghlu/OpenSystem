/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
#include "PseudoClassChangeInvalidation.h"

#include "ElementChildIteratorInlines.h"
#include "ElementRareData.h"
#include "StyleInvalidationFunctions.h"

namespace WebCore {
namespace Style {

Vector<PseudoClassInvalidationKey, 4> makePseudoClassInvalidationKeys(CSSSelector::PseudoClass pseudoClass, const Element& element)
{
    Vector<PseudoClassInvalidationKey, 4> keys;

    if (!element.idForStyleResolution().isEmpty())
        keys.append(makePseudoClassInvalidationKey(pseudoClass, InvalidationKeyType::Id, element.idForStyleResolution()));

    if (element.hasClass()) {
        keys.appendContainerWithMapping(element.classNames(), [&](auto& className) {
            return makePseudoClassInvalidationKey(pseudoClass, InvalidationKeyType::Class, className);
        });
    }

    keys.append(makePseudoClassInvalidationKey(pseudoClass, InvalidationKeyType::Tag, element.localNameLowercase()));
    keys.append(makePseudoClassInvalidationKey(pseudoClass, InvalidationKeyType::Universal));

    return keys;
};

void PseudoClassChangeInvalidation::computeInvalidation(CSSSelector::PseudoClass pseudoClass, Value value, InvalidationScope invalidationScope)
{
    bool shouldInvalidateCurrent = false;
    bool mayAffectStyleInShadowTree = false;

    traverseRuleFeatures(m_element, [&] (const RuleFeatureSet& features, bool mayAffectShadowTree) {
        if (mayAffectShadowTree && features.pseudoClasses.contains(pseudoClass))
            mayAffectStyleInShadowTree = true;
        if (m_element.shadowRoot() && features.pseudoClassesAffectingHost.contains(pseudoClass))
            shouldInvalidateCurrent = true;
    });

    if (mayAffectStyleInShadowTree) {
        // FIXME: We should do fine-grained invalidation for shadow tree.
        m_element.invalidateStyleForSubtree();
    }

    if (shouldInvalidateCurrent)
        m_element.invalidateStyle();

    for (auto& key : makePseudoClassInvalidationKeys(pseudoClass, m_element))
        collectRuleSets(key, value, invalidationScope);
}

void PseudoClassChangeInvalidation::collectRuleSets(const PseudoClassInvalidationKey& key, Value value, InvalidationScope invalidationScope)
{
    auto collect = [&](auto& ruleSets, std::optional<MatchElement> onlyMatchElement = { }) {
        auto* invalidationRuleSets = ruleSets.pseudoClassInvalidationRuleSets(key);
        if (!invalidationRuleSets)
            return;

        for (auto& invalidationRuleSet : *invalidationRuleSets) {
            if (onlyMatchElement && invalidationRuleSet.matchElement != onlyMatchElement)
                continue;

            // For focus/hover we flip the whole ancestor chain. We only need to do deep invalidation traversal in the change root.
            auto shouldInvalidate = [&] {
                bool invalidatesAllDescendants = invalidationRuleSet.matchElement == MatchElement::Ancestor && isUniversalInvalidation(key);
                switch (invalidationScope) {
                case InvalidationScope::All:
                    return true;
                case InvalidationScope::SelfChildrenAndSiblings:
                    return !invalidatesAllDescendants;
                case InvalidationScope::Descendants:
                    return invalidatesAllDescendants;
                }
                ASSERT_NOT_REACHED();
                return true;
            }();
            if (!shouldInvalidate)
                continue;

            if (value == Value::Any) {
                Invalidator::addToMatchElementRuleSets(m_beforeChangeRuleSets, invalidationRuleSet);
                Invalidator::addToMatchElementRuleSets(m_afterChangeRuleSets, invalidationRuleSet);
                continue;
            }

            bool invalidateBeforeChange = invalidationRuleSet.isNegation == IsNegation::Yes ? value == Value::True : value == Value::False;
            if (invalidateBeforeChange)
                Invalidator::addToMatchElementRuleSets(m_beforeChangeRuleSets, invalidationRuleSet);
            else
                Invalidator::addToMatchElementRuleSets(m_afterChangeRuleSets, invalidationRuleSet);
        }
    };

    collect(m_element.styleResolver().ruleSets());

    if (auto* shadowRoot = m_element.shadowRoot())
        collect(shadowRoot->styleScope().resolver().ruleSets(), MatchElement::Host);
}

void PseudoClassChangeInvalidation::invalidateBeforeChange()
{
    Invalidator::invalidateWithMatchElementRuleSets(m_element, m_beforeChangeRuleSets);
}

void PseudoClassChangeInvalidation::invalidateAfterChange()
{
    Invalidator::invalidateWithMatchElementRuleSets(m_element, m_afterChangeRuleSets);
}


}
}
