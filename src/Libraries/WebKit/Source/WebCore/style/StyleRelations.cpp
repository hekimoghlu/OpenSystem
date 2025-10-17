/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
#include "StyleRelations.h"

#include "Element.h"
#include "NodeRenderStyle.h"
#include "RenderStyle.h"
#include "StyleUpdate.h"

namespace WebCore {
namespace Style {

std::unique_ptr<Relations> commitRelationsToRenderStyle(RenderStyle& style, const Element& element, const Relations& relations)
{
    if (!relations.isEmpty())
        style.setUnique();

    std::unique_ptr<Relations> remainingRelations;

    auto appendStyleRelation = [&remainingRelations] (const Relation& relation) {
        if (!remainingRelations)
            remainingRelations = makeUnique<Relations>();
        remainingRelations->append(relation);
    };

    for (auto& relation : relations) {
        if (relation.element != &element) {
            appendStyleRelation(relation);
            continue;
        }
        switch (relation.type) {
        case Relation::AffectedByEmpty:
            style.setEmptyState(relation.value);
            appendStyleRelation(relation);
            break;
        case Relation::FirstChild:
            style.setFirstChildState();
            break;
        case Relation::LastChild:
            style.setLastChildState();
            break;
        case Relation::Unique:
            break;
        case Relation::AffectedByPreviousSibling:
        case Relation::DescendantsAffectedByPreviousSibling:
        case Relation::AffectsNextSibling:
        case Relation::ChildrenAffectedByForwardPositionalRules:
        case Relation::DescendantsAffectedByForwardPositionalRules:
        case Relation::ChildrenAffectedByBackwardPositionalRules:
        case Relation::DescendantsAffectedByBackwardPositionalRules:
        case Relation::ChildrenAffectedByFirstChildRules:
        case Relation::ChildrenAffectedByLastChildRules:
        case Relation::NthChildIndex:
        case Relation::AffectedByHasWithPositionalPseudoClass:
            appendStyleRelation(relation);
            break;
        }
    }
    return remainingRelations;
}

void commitRelations(std::unique_ptr<Relations> relations, Update& update)
{
    if (!relations)
        return;
    for (auto& relation : *relations) {
        auto& element = const_cast<Element&>(*relation.element);
        switch (relation.type) {
        case Relation::AffectedByEmpty:
            element.setStyleAffectedByEmpty();
            break;
        case Relation::AffectedByPreviousSibling:
            element.setStyleIsAffectedByPreviousSibling();
            break;
        case Relation::DescendantsAffectedByPreviousSibling:
            element.setDescendantsAffectedByPreviousSibling();
            break;
        case Relation::AffectsNextSibling: {
            auto* sibling = &element;
            for (unsigned i = 0; i < relation.value && sibling; ++i, sibling = sibling->nextElementSibling())
                sibling->setAffectsNextSiblingElementStyle();
            break;
        }
        case Relation::ChildrenAffectedByForwardPositionalRules:
            element.setChildrenAffectedByForwardPositionalRules();
            break;
        case Relation::DescendantsAffectedByForwardPositionalRules:
            element.setDescendantsAffectedByForwardPositionalRules();
            break;
        case Relation::ChildrenAffectedByBackwardPositionalRules:
            element.setChildrenAffectedByBackwardPositionalRules();
            break;
        case Relation::DescendantsAffectedByBackwardPositionalRules:
            element.setDescendantsAffectedByBackwardPositionalRules();
            break;
        case Relation::ChildrenAffectedByFirstChildRules:
            element.setChildrenAffectedByFirstChildRules();
            break;
        case Relation::ChildrenAffectedByLastChildRules:
            element.setChildrenAffectedByLastChildRules();
            break;
        case Relation::AffectedByHasWithPositionalPseudoClass:
            element.setAffectedByHasWithPositionalPseudoClass();
            break;
        case Relation::FirstChild:
            if (auto* style = update.elementStyle(element))
                style->setFirstChildState();
            break;
        case Relation::LastChild:
            if (auto* style = update.elementStyle(element))
                style->setLastChildState();
            break;
        case Relation::NthChildIndex:
            if (auto* style = update.elementStyle(element))
                style->setUnique();
            element.setChildIndex(relation.value);
            break;
        case Relation::Unique:
            if (auto* style = update.elementStyle(element))
                style->setUnique();
            break;
        }
    }
}

void copyRelations(RenderStyle& to, const RenderStyle& from)
{
    if (from.emptyState())
        to.setEmptyState(true);
    if (from.firstChildState())
        to.setFirstChildState();
    if (from.lastChildState())
        to.setLastChildState();
    if (from.unique())
        to.setUnique();
}

}
}
