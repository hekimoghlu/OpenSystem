/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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

#include <wtf/Forward.h>

namespace WebCore {

class Element;
class RenderStyle;

namespace Style {

class Update;

struct Relation {
    enum Type {
        AffectedByEmpty,
        AffectedByPreviousSibling,
        DescendantsAffectedByPreviousSibling,
        // For AffectsNextSibling 'value' tells how many element siblings to mark starting with 'element'.
        AffectsNextSibling,
        ChildrenAffectedByForwardPositionalRules,
        DescendantsAffectedByForwardPositionalRules,
        ChildrenAffectedByBackwardPositionalRules,
        DescendantsAffectedByBackwardPositionalRules,
        ChildrenAffectedByFirstChildRules,
        ChildrenAffectedByLastChildRules,
        FirstChild,
        LastChild,
        NthChildIndex,
        Unique,
        AffectedByHasWithPositionalPseudoClass,
    };
    const Element* element;
    Type type;
    unsigned value;

    Relation(const Element& element, Type type, unsigned value = 1)
        : element(&element)
        , type(type)
        , value(value)
    { }
};

using Relations = Vector<Relation, 8>;

std::unique_ptr<Relations> commitRelationsToRenderStyle(RenderStyle&, const Element&, const Relations&);
void copyRelations(RenderStyle&, const RenderStyle&);
void commitRelations(std::unique_ptr<Relations>, Update&);

}
}
