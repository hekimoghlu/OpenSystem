/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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
#include "ClassChangeInvalidation.h"

#include "ElementChildIteratorInlines.h"
#include "ElementRareData.h"
#include "SpaceSplitString.h"
#include "StyleInvalidationFunctions.h"
#include <wtf/BitVector.h>

namespace WebCore {
namespace Style {

enum class ClassChangeType : bool { Add, Remove };

struct ClassChange {
    AtomStringImpl* className { };
    ClassChangeType type;
};

constexpr size_t classChangeVectorInlineCapacity = 4;
using ClassChangeVector = Vector<ClassChange, classChangeVectorInlineCapacity>;

static ClassChangeVector collectClasses(const SpaceSplitString& classes, ClassChangeType changeType)
{
    return WTF::map<classChangeVectorInlineCapacity>(classes, [changeType](auto& className) {
        return ClassChange { className.impl(), changeType };
    });
}

static ClassChangeVector computeClassChanges(const SpaceSplitString& oldClasses, const SpaceSplitString& newClasses)
{
    unsigned oldSize = oldClasses.size();

    if (!oldSize)
        return collectClasses(newClasses, ClassChangeType::Add);
    if (newClasses.isEmpty())
        return collectClasses(oldClasses, ClassChangeType::Remove);

    ClassChangeVector changedClasses;

    BitVector remainingClassBits;
    remainingClassBits.ensureSize(oldSize);
    // Class vectors tend to be very short. This is faster than using a hash table.
    for (auto& newClass : newClasses) {
        bool foundFromBoth = false;
        for (unsigned i = 0; i < oldSize; ++i) {
            if (newClass == oldClasses[i]) {
                remainingClassBits.quickSet(i);
                foundFromBoth = true;
            }
        }
        if (foundFromBoth)
            continue;
        changedClasses.append({ newClass.impl(), ClassChangeType::Add });
    }
    for (unsigned i = 0; i < oldSize; ++i) {
        // If the bit is not set the corresponding class has been removed.
        if (remainingClassBits.quickGet(i))
            continue;
        changedClasses.append({ oldClasses[i].impl(), ClassChangeType::Remove });
    }

    return changedClasses;
}

void ClassChangeInvalidation::computeInvalidation(const SpaceSplitString& oldClasses, const SpaceSplitString& newClasses)
{
    auto classChanges = computeClassChanges(oldClasses, newClasses);

    bool shouldInvalidateCurrent = false;
    bool mayAffectStyleInShadowTree = false;

    traverseRuleFeatures(m_element, [&] (const RuleFeatureSet& features, bool mayAffectShadowTree) {
        for (auto& classChange : classChanges) {
            if (mayAffectShadowTree && features.classRules.contains(classChange.className))
                mayAffectStyleInShadowTree = true;
            if (features.classesAffectingHost.contains(classChange.className))
                shouldInvalidateCurrent = true;
        }
    });

    if (mayAffectStyleInShadowTree) {
        // FIXME: We should do fine-grained invalidation for shadow tree.
        m_element.invalidateStyleForSubtree();
    }

    if (shouldInvalidateCurrent)
        m_element.invalidateStyle();

    auto invalidateBeforeAndAfterChange = [](MatchElement matchElement) {
        switch (matchElement) {
        case MatchElement::AnySibling:
        case MatchElement::ParentAnySibling:
        case MatchElement::AncestorAnySibling:
        case MatchElement::HasAnySibling:
        case MatchElement::HasNonSubject:
        case MatchElement::HasScopeBreaking:
            return true;
        case MatchElement::Subject:
        case MatchElement::Parent:
        case MatchElement::Ancestor:
        case MatchElement::DirectSibling:
        case MatchElement::IndirectSibling:
        case MatchElement::ParentSibling:
        case MatchElement::AncestorSibling:
        case MatchElement::HasChild:
        case MatchElement::HasDescendant:
        case MatchElement::HasSibling:
        case MatchElement::HasSiblingDescendant:
        case MatchElement::Host:
        case MatchElement::HostChild:
            return false;
        }
        ASSERT_NOT_REACHED();
        return false;
    };

    auto invalidateBeforeChange = [&](ClassChangeType type, IsNegation isNegation, MatchElement matchElement) {
        if (invalidateBeforeAndAfterChange(matchElement))
            return true;
        return type == ClassChangeType::Remove ? isNegation == IsNegation::No : isNegation == IsNegation::Yes;
    };

    auto invalidateAfterChange = [&](ClassChangeType type, IsNegation isNegation, MatchElement matchElement) {
        if (invalidateBeforeAndAfterChange(matchElement))
            return true;
        return type == ClassChangeType::Add ? isNegation == IsNegation::No : isNegation == IsNegation::Yes;
    };

    auto collect = [&](auto& ruleSets, std::optional<MatchElement> onlyMatchElement = { }) {
        for (auto& classChange : classChanges) {
            if (auto* invalidationRuleSets = ruleSets.classInvalidationRuleSets(classChange.className)) {
                for (auto& invalidationRuleSet : *invalidationRuleSets) {
                    if (onlyMatchElement && invalidationRuleSet.matchElement != onlyMatchElement)
                        continue;

                    if (invalidateBeforeChange(classChange.type, invalidationRuleSet.isNegation, invalidationRuleSet.matchElement))
                        Invalidator::addToMatchElementRuleSets(m_beforeChangeRuleSets, invalidationRuleSet);
                    if (invalidateAfterChange(classChange.type, invalidationRuleSet.isNegation, invalidationRuleSet.matchElement))
                        Invalidator::addToMatchElementRuleSets(m_afterChangeRuleSets, invalidationRuleSet);
                }
            }
        }
    };

    collect(m_element.styleResolver().ruleSets());

    if (auto* shadowRoot = m_element.shadowRoot())
        collect(shadowRoot->styleScope().resolver().ruleSets(), MatchElement::Host);
}

void ClassChangeInvalidation::invalidateBeforeChange()
{
    Invalidator::invalidateWithMatchElementRuleSets(m_element, m_beforeChangeRuleSets);
}

void ClassChangeInvalidation::invalidateAfterChange()
{
    Invalidator::invalidateWithMatchElementRuleSets(m_element, m_afterChangeRuleSets);
}

}
}
