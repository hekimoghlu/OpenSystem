/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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
#include "HTMLTableSectionElement.h"

#include "ElementChildIteratorInlines.h"
#include "GenericCachedHTMLCollection.h"
#include "HTMLNames.h"
#include "HTMLTableRowElement.h"
#include "HTMLTableElement.h"
#include "NodeList.h"
#include "NodeRareData.h"
#include "Text.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLTableSectionElement);

using namespace HTMLNames;

inline HTMLTableSectionElement::HTMLTableSectionElement(const QualifiedName& tagName, Document& document)
    : HTMLTablePartElement(tagName, document)
{
}

Ref<HTMLTableSectionElement> HTMLTableSectionElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLTableSectionElement(tagName, document));
}

const MutableStyleProperties* HTMLTableSectionElement::additionalPresentationalHintStyle() const
{
    auto table = findParentTable();
    if (!table)
        return nullptr;
    return table->additionalGroupStyle(true);
}

ExceptionOr<Ref<HTMLTableRowElement>> HTMLTableSectionElement::insertRow(int index)
{
    if (index < -1)
        return Exception { ExceptionCode::IndexSizeError };
    auto children = rows();
    int numRows = children->length();
    if (index > numRows)
        return Exception { ExceptionCode::IndexSizeError };
    auto row = HTMLTableRowElement::create(trTag, document());
    ExceptionOr<void> result;
    if (numRows == index || index == -1)
        result = appendChild(row);
    else
        result = insertBefore(row, children->item(index));
    if (result.hasException())
        return result.releaseException();
    return row;
}

ExceptionOr<void> HTMLTableSectionElement::deleteRow(int index)
{
    auto children = rows();
    int numRows = children->length();
    if (index == -1) {
        if (!numRows)
            return { };
        index = numRows - 1;
    }
    if (index < 0 || index >= numRows)
        return Exception { ExceptionCode::IndexSizeError };
    return removeChild(*children->item(index));
}

int HTMLTableSectionElement::numRows() const
{
    auto rows = childrenOfType<HTMLTableRowElement>(*this);
    return std::distance(rows.begin(), { });
}

Ref<HTMLCollection> HTMLTableSectionElement::rows()
{
    return ensureRareData().ensureNodeLists().addCachedCollection<GenericCachedHTMLCollection<CollectionTypeTraits<CollectionType::TSectionRows>::traversalType>>(*this, CollectionType::TSectionRows);
}

}
