/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
#include "HTMLTableRowsCollection.h"

#include "CachedHTMLCollectionInlines.h"
#include "ElementIterator.h"
#include "ElementTraversal.h"
#include "HTMLNames.h"
#include "HTMLTableElement.h"
#include "HTMLTableRowElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace HTMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLTableRowsCollection);

static inline void assertRowIsInTable(HTMLTableElement& table, HTMLTableRowElement* row)
{
#if ASSERT_ENABLED
    UNUSED_PARAM(table);
    UNUSED_PARAM(row);
#else // not ASSERT_ENABLED
    if (!row)
        return;
    if (row->parentNode() == &table)
        return;
    ASSERT(row->parentNode());
    ASSERT(row->parentNode()->hasTagName(theadTag) || row->parentNode()->hasTagName(tbodyTag) || row->parentNode()->hasTagName(tfootTag));
    ASSERT(row->parentNode()->parentNode() == &table);
#endif // not ASSERT_ENABLED
}

static inline bool isInSection(HTMLTableRowElement& row, const HTMLQualifiedName& sectionTag)
{
    // Because we know that the parent is a table or a section, it's safe to cast it to an HTMLElement
    // giving us access to the faster hasTagName overload from that class.
    return downcast<HTMLElement>(*row.parentNode()).hasTagName(sectionTag);
}

HTMLTableRowElement* HTMLTableRowsCollection::rowAfter(HTMLTableElement& table, HTMLTableRowElement* previous)
{
    // The HTMLCollection caching mechanism, along with the code in this class, will guarantee that the previous row
    // is an immediate child of either the table, or a table section that is itself an immediate child of the table.
    assertRowIsInTable(table, previous);

    // Start by looking for the next row in this section. Continue only if there is none.
    if (previous && previous->parentNode() != &table) {
        auto childRows = childrenOfType<HTMLTableRowElement>(*previous->parentNode());
        auto row = childRows.beginAt(*previous);
        if (++row != childRows.end())
            return &*row;
    }

    RefPtr<Element> child;

    // If still looking at head sections, find the first row in the next head section.
    if (!previous)
        child = ElementTraversal::firstChild(table);
    else if (isInSection(*previous, theadTag))
        child = ElementTraversal::nextSibling(*previous->parentNode());
    for (; child; child = ElementTraversal::nextSibling(*child)) {
        if (child->hasTagName(theadTag)) {
            if (auto row = childrenOfType<HTMLTableRowElement>(*child).first())
                return row;
        }
    }

    // If still looking at top level and bodies, find the next row in top level or the first in the next body section.
    if (!previous || isInSection(*previous, theadTag))
        child = ElementTraversal::firstChild(table);
    else if (previous->parentNode() == &table)
        child = ElementTraversal::nextSibling(*previous);
    else if (isInSection(*previous, tbodyTag))
        child = ElementTraversal::nextSibling(*previous->parentNode());
    for (; child; child = ElementTraversal::nextSibling(*child)) {
        if (auto* row = dynamicDowncast<HTMLTableRowElement>(*child))
            return row;
        if (child->hasTagName(tbodyTag)) {
            if (auto row = childrenOfType<HTMLTableRowElement>(*child).first())
                return row;
        }
    }

    // Find the first row in the next foot section.
    if (!previous || !isInSection(*previous, tfootTag))
        child = ElementTraversal::firstChild(table);
    else
        child = ElementTraversal::nextSibling(*previous->parentNode());
    for (; child; child = ElementTraversal::nextSibling(*child)) {
        if (child->hasTagName(tfootTag)) {
            if (auto row = childrenOfType<HTMLTableRowElement>(*child).first())
                return row;
        }
    }

    return nullptr;
}

HTMLTableRowElement* HTMLTableRowsCollection::lastRow(HTMLTableElement& table)
{
    for (auto* child = ElementTraversal::lastChild(table); child; child = ElementTraversal::previousSibling(*child)) {
        if (child->hasTagName(tfootTag)) {
            if (auto* row = childrenOfType<HTMLTableRowElement>(*child).last())
                return row;
        }
    }

    for (auto* child = ElementTraversal::lastChild(table); child; child = ElementTraversal::previousSibling(*child)) {
        if (auto* row = dynamicDowncast<HTMLTableRowElement>(*child))
            return row;
        if (child->hasTagName(tbodyTag)) {
            if (auto* row = childrenOfType<HTMLTableRowElement>(*child).last())
                return row;
        }
    }

    for (auto* child = ElementTraversal::lastChild(table); child; child = ElementTraversal::previousSibling(*child)) {
        if (child->hasTagName(theadTag)) {
            if (auto* row = childrenOfType<HTMLTableRowElement>(*child).last())
                return row;
        }
    }

    return nullptr;
}

HTMLTableRowsCollection::HTMLTableRowsCollection(HTMLTableElement& table)
    : CachedHTMLCollection(table, CollectionType::TableRows)
{
}

Ref<HTMLTableRowsCollection> HTMLTableRowsCollection::create(HTMLTableElement& table, CollectionType type)
{
    ASSERT_UNUSED(type, type == CollectionType::TableRows);
    return adoptRef(*new HTMLTableRowsCollection(table));
}

Element* HTMLTableRowsCollection::customElementAfter(Element* previous) const
{
    return rowAfter(const_cast<HTMLTableElement&>(tableElement()), downcast<HTMLTableRowElement>(previous));
}

}
