/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#include "ChildChangeInvalidation.h"

#include "ElementTraversal.h"
#include "NodeRenderStyle.h"
#include "PseudoClassChangeInvalidation.h"
#include "RenderElement.h"
#include "ShadowRoot.h"
#include "SlotAssignment.h"
#include "StyleResolver.h"
#include "StyleScopeRuleSets.h"
#include "TypedElementDescendantIteratorInlines.h"

namespace WebCore::Style {

void ChildChangeInvalidation::invalidateForChangedElement(Element& changedElement, MatchingHasSelectors& matchingHasSelectors, ChangedElementRelation changedElementRelation)
{
    auto& ruleSets = parentElement().styleResolver().ruleSets();

    Invalidator::MatchElementRuleSets matchElementRuleSets;

    bool isChild = changedElement.parentElement() == &parentElement();

    auto canAffectElementsWithStyle = [&](MatchElement matchElement) {
        switch (matchElement) {
        case MatchElement::HasSibling:
        case MatchElement::HasChild:
            return isChild;
        case MatchElement::HasDescendant:
        case MatchElement::HasSiblingDescendant:
        case MatchElement::HasNonSubject:
        case MatchElement::HasScopeBreaking:
            return true;
        default:
            ASSERT_NOT_REACHED();
            return false;
        }
    };

    bool isFirst = isChild && m_childChange.previousSiblingElement == changedElement.previousElementSibling() && changedElementRelation == ChangedElementRelation::SelfOrDescendant;

    auto hasMatchingInvalidationSelector = [&](auto& invalidationRuleSet) {
        SelectorChecker selectorChecker(changedElement.document());
        SelectorChecker::CheckingContext checkingContext(SelectorChecker::Mode::CollectingRulesIgnoringVirtualPseudoElements);
        checkingContext.matchesAllHasScopes = true;

        for (auto* selector : invalidationRuleSet.invalidationSelectors) {
            if (isFirst && invalidationRuleSet.isNegation == IsNegation::No) {
                // If this :has() matches ignoring this mutation, nothing actually changes and we don't need to invalidate.
                // FIXME: We could cache this state across invalidations instead of just testing a single sibling.
                auto* sibling = m_childChange.previousSiblingElement ? m_childChange.previousSiblingElement : m_childChange.nextSiblingElement;
                if (sibling && selectorChecker.match(*selector, *sibling, checkingContext)) {
                    matchingHasSelectors.add(selector);
                    continue;
                }
            }

            if (matchingHasSelectors.contains(selector))
                continue;

            if (selectorChecker.match(*selector, changedElement, checkingContext)) {
                matchingHasSelectors.add(selector);
                return true;
            }
        }
        return false;
    };

    auto addHasInvalidation = [&](const Vector<InvalidationRuleSet>* invalidationRuleSets) {
        if (!invalidationRuleSets)
            return;
        for (auto& invalidationRuleSet : *invalidationRuleSets) {
            if (!canAffectElementsWithStyle(invalidationRuleSet.matchElement))
                continue;
            if (!hasMatchingInvalidationSelector(invalidationRuleSet))
                continue;
            Invalidator::addToMatchElementRuleSetsRespectingNegation(matchElementRuleSets, invalidationRuleSet);
        }
    };

    for (auto key : makePseudoClassInvalidationKeys(CSSSelector::PseudoClass::Has, changedElement))
        addHasInvalidation(ruleSets.hasPseudoClassInvalidationRuleSets(key));

    Invalidator::invalidateWithMatchElementRuleSets(changedElement, matchElementRuleSets);
}

void ChildChangeInvalidation::invalidateForChangeOutsideHasScope()
{
    // FIXME: This is a performance footgun. Any mutation will trigger a full document traversal.
    if (auto* invalidationRuleSet = parentElement().styleResolver().ruleSets().scopeBreakingHasPseudoClassInvalidationRuleSet())
        Invalidator::invalidateWithScopeBreakingHasPseudoClassRuleSet(parentElement(), invalidationRuleSet);
}

void ChildChangeInvalidation::invalidateForHasBeforeMutation()
{
    ASSERT(m_needsHasInvalidation);

    invalidateForChangeOutsideHasScope();

    MatchingHasSelectors matchingHasSelectors;

    traverseRemovedElements([&](auto& changedElement) {
        invalidateForChangedElement(changedElement, matchingHasSelectors, ChangedElementRelation::SelfOrDescendant);
    });

    // :empty is affected by text changes.
    if (m_childChange.type == ContainerNode::ChildChange::Type::TextRemoved || m_childChange.type == ContainerNode::ChildChange::Type::AllChildrenRemoved)
        invalidateForChangedElement(parentElement(), matchingHasSelectors, ChangedElementRelation::SelfOrDescendant);

    auto firstChildStateWillStopMatching = [&] {
        if (!m_childChange.nextSiblingElement)
            return false;

        if (!parentElement().childrenAffectedByFirstChildRules())
            return false;

        if (m_childChange.isInsertion() && !m_childChange.nextSiblingElement->previousElementSibling())
            return true;

        return false;
    };

    auto lastChildStateWillStopMatching = [&] {
        if (!m_childChange.previousSiblingElement)
            return false;

        if (!parentElement().childrenAffectedByLastChildRules())
            return false;

        if (m_childChange.isInsertion() && !m_childChange.previousSiblingElement->nextElementSibling())
            return true;

        return false;
    };

    if (parentElement().affectedByHasWithPositionalPseudoClass()) {
        traverseRemainingExistingSiblings([&](auto& changedElement) {
            invalidateForChangedElement(changedElement, matchingHasSelectors, ChangedElementRelation::Sibling);
        });
    } else {
        if (firstChildStateWillStopMatching())
            invalidateForChangedElement(*m_childChange.nextSiblingElement, matchingHasSelectors, ChangedElementRelation::Sibling);

        if (lastChildStateWillStopMatching())
            invalidateForChangedElement(*m_childChange.previousSiblingElement, matchingHasSelectors, ChangedElementRelation::Sibling);
    }
}

void ChildChangeInvalidation::invalidateForHasAfterMutation()
{
    ASSERT(m_needsHasInvalidation);

    invalidateForChangeOutsideHasScope();

    MatchingHasSelectors matchingHasSelectors;

    traverseAddedElements([&](auto& changedElement) {
        invalidateForChangedElement(changedElement, matchingHasSelectors, ChangedElementRelation::SelfOrDescendant);
    });

    // :empty is affected by text changes.
    if (m_childChange.type == ContainerNode::ChildChange::Type::TextInserted && m_wasEmpty)
        invalidateForChangedElement(parentElement(), matchingHasSelectors, ChangedElementRelation::SelfOrDescendant);

    auto firstChildStateWillStartMatching = [&](Element* elementAfterChange) {
        if (!elementAfterChange)
            return false;

        if (!parentElement().childrenAffectedByFirstChildRules())
            return false;

        if (!m_childChange.isInsertion() && !elementAfterChange->previousElementSibling())
            return true;

        return false;
    };

    auto lastChildStateWillStartMatching = [&](Element* elementBeforeChange) {
        if (!elementBeforeChange)
            return false;

        if (!parentElement().childrenAffectedByLastChildRules())
            return false;

        if (!m_childChange.isInsertion() && !elementBeforeChange->nextElementSibling())
            return true;

        return false;
    };

    if (parentElement().affectedByHasWithPositionalPseudoClass()) {
        traverseRemainingExistingSiblings([&](auto& changedElement) {
            invalidateForChangedElement(changedElement, matchingHasSelectors, ChangedElementRelation::Sibling);
        });
    } else {
        if (firstChildStateWillStartMatching(m_childChange.nextSiblingElement))
            invalidateForChangedElement(*m_childChange.nextSiblingElement, matchingHasSelectors, ChangedElementRelation::Sibling);

        if (lastChildStateWillStartMatching(m_childChange.previousSiblingElement))
            invalidateForChangedElement(*m_childChange.previousSiblingElement, matchingHasSelectors, ChangedElementRelation::Sibling);
    }
}

static bool needsDescendantTraversal(const RuleFeatureSet& features)
{
    return features.usesMatchElement(MatchElement::HasNonSubject)
        || features.usesMatchElement(MatchElement::HasScopeBreaking)
        || features.usesMatchElement(MatchElement::HasDescendant)
        || features.usesMatchElement(MatchElement::HasSiblingDescendant);
};

template<typename Function>
void ChildChangeInvalidation::traverseRemovedElements(Function&& function)
{
    if (m_childChange.isInsertion() && m_childChange.type != ContainerNode::ChildChange::Type::AllChildrenReplaced)
        return;

    auto& features = parentElement().styleResolver().ruleSets().features();
    bool needsDescendantTraversal = Style::needsDescendantTraversal(features);

    auto* firstToRemove = m_childChange.previousSiblingElement ? m_childChange.previousSiblingElement->nextElementSibling() : parentElement().firstElementChild();

    for (auto* toRemove = firstToRemove; toRemove != m_childChange.nextSiblingElement; toRemove = toRemove->nextElementSibling()) {
        function(*toRemove);

        if (!needsDescendantTraversal)
            continue;

        for (auto& descendant : descendantsOfType<Element>(*toRemove))
            function(descendant);
    }
}

template<typename Function>
void ChildChangeInvalidation::traverseAddedElements(Function&& function)
{
    if (!m_childChange.isInsertion())
        return;

    auto* newElement = [&] {
        auto* previous = m_childChange.previousSiblingElement;
        auto* candidate = previous ? ElementTraversal::nextSibling(*previous) : ElementTraversal::firstChild(parentElement());
        if (candidate == m_childChange.nextSiblingElement)
            candidate = nullptr;
        return candidate;
    }();

    if (!newElement)
        return;

    function(*newElement);

    auto& features = parentElement().styleResolver().ruleSets().features();
    if (!needsDescendantTraversal(features))
        return;

    for (auto& descendant : descendantsOfType<Element>(*newElement))
        function(descendant);
}

template<typename Function>
void ChildChangeInvalidation::traverseRemainingExistingSiblings(Function&& function)
{
    if (m_childChange.isInsertion() && m_childChange.type == ContainerNode::ChildChange::Type::AllChildrenReplaced)
        return;

    for (auto* child = m_childChange.previousSiblingElement; child; child = child->previousElementSibling())
        function(*child);

    for (auto* child = m_childChange.nextSiblingElement; child; child = child->nextElementSibling())
        function(*child);
}

static void checkForEmptyStyleChange(Element& element)
{
    if (!element.styleAffectedByEmpty())
        return;

    auto* style = element.renderStyle();
    if (!style || (!style->emptyState() || element.hasChildNodes()))
        element.invalidateStyleForSubtree();
}

static void invalidateForForwardPositionalRules(Element& parent, Element* elementAfterChange)
{
    bool childrenAffected = parent.childrenAffectedByForwardPositionalRules();
    bool descendantsAffected = parent.descendantsAffectedByForwardPositionalRules();

    if (!childrenAffected && !descendantsAffected)
        return;

    for (auto* sibling = elementAfterChange; sibling; sibling = sibling->nextElementSibling()) {
        if (childrenAffected)
            sibling->invalidateStyleInternal();
        if (descendantsAffected) {
            for (auto* siblingChild = sibling->firstElementChild(); siblingChild; siblingChild = siblingChild->nextElementSibling())
                siblingChild->invalidateStyleForSubtreeInternal();
        }
    }
}

static void invalidateForBackwardPositionalRules(Element& parent, Element* elementBeforeChange)
{
    bool childrenAffected = parent.childrenAffectedByBackwardPositionalRules();
    bool descendantsAffected = parent.descendantsAffectedByBackwardPositionalRules();

    if (!childrenAffected && !descendantsAffected)
        return;

    for (auto* sibling = elementBeforeChange; sibling; sibling = sibling->previousElementSibling()) {
        if (childrenAffected)
            sibling->invalidateStyleInternal();
        if (descendantsAffected) {
            for (auto* siblingChild = sibling->firstElementChild(); siblingChild; siblingChild = siblingChild->nextElementSibling())
                siblingChild->invalidateStyleForSubtreeInternal();
        }
    }
}

static void invalidateForFirstChildState(Element& child, bool state)
{
    auto* style = child.renderStyle();
    if (!style || style->firstChildState() == state)
        child.invalidateStyleForSubtreeInternal();
}

static void invalidateForLastChildState(Element& child, bool state)
{
    auto* style = child.renderStyle();
    if (!style || style->lastChildState() == state)
        child.invalidateStyleForSubtreeInternal();
}

void ChildChangeInvalidation::invalidateAfterChange()
{
    checkForEmptyStyleChange(parentElement());

    if (m_childChange.source == ContainerNode::ChildChange::Source::Parser)
        return;

    checkForSiblingStyleChanges();
}

void ChildChangeInvalidation::invalidateAfterFinishedParsingChildren(Element& parent)
{
    if (!parent.needsStyleInvalidation())
        return;

    checkForEmptyStyleChange(parent);

    auto* lastChildElement = ElementTraversal::lastChild(parent);
    if (!lastChildElement)
        return;

    if (parent.childrenAffectedByLastChildRules())
        invalidateForLastChildState(*lastChildElement, false);

    invalidateForBackwardPositionalRules(parent, lastChildElement);
}

void ChildChangeInvalidation::checkForSiblingStyleChanges()
{
    auto& parent = parentElement();
    auto* elementBeforeChange = m_childChange.previousSiblingElement;
    auto* elementAfterChange = m_childChange.nextSiblingElement;

    // :first-child. In the parser callback case, we don't have to check anything, since we were right the first time.
    // In the DOM case, we only need to do something if |afterChange| is not 0.
    // |afterChange| is 0 in the parser case, so it works out that we'll skip this block.
    if (parent.childrenAffectedByFirstChildRules() && elementAfterChange) {
        // Find our new first child.
        RefPtr<Element> newFirstElement = ElementTraversal::firstChild(parent);

        // This is the insert/append case.
        if (newFirstElement != elementAfterChange)
            invalidateForFirstChildState(*elementAfterChange, true);

        // We also have to handle node removal.
        if (m_childChange.type == ContainerNode::ChildChange::Type::ElementRemoved && newFirstElement == elementAfterChange)
            invalidateForFirstChildState(*newFirstElement, false);
    }

    // :last-child. In the parser callback case, we don't have to check anything, since we were right the first time.
    // In the DOM case, we only need to do something if |afterChange| is not 0.
    if (parent.childrenAffectedByLastChildRules() && elementBeforeChange) {
        // Find our new last child.
        RefPtr<Element> newLastElement = ElementTraversal::lastChild(parent);

        if (newLastElement != elementBeforeChange)
            invalidateForLastChildState(*elementBeforeChange, true);

        // We also have to handle node removal.
        if (m_childChange.type == ContainerNode::ChildChange::Type::ElementRemoved && newLastElement == elementBeforeChange)
            invalidateForLastChildState(*newLastElement, false);
    }

    invalidateForSiblingCombinators(elementAfterChange);

    invalidateForForwardPositionalRules(parent, elementAfterChange);
    invalidateForBackwardPositionalRules(parent, elementBeforeChange);
}

}
