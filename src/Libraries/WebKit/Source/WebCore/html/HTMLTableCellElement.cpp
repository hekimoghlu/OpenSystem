/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#include "HTMLTableCellElement.h"

#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "ElementInlines.h"
#include "HTMLNames.h"
#include "HTMLParserIdioms.h"
#include "HTMLTableElement.h"
#include "NodeName.h"
#include "RenderTableCell.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLTableCellElement);

using namespace HTMLNames;

Ref<HTMLTableCellElement> HTMLTableCellElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLTableCellElement(tagName, document));
}

HTMLTableCellElement::HTMLTableCellElement(const QualifiedName& tagName, Document& document)
    : HTMLTablePartElement(tagName, document)
{
    ASSERT(hasLocalName(thTag->localName()) || hasLocalName(tdTag->localName()));
}

unsigned HTMLTableCellElement::colSpan() const
{
    return clampHTMLNonNegativeIntegerToRange(attributeWithoutSynchronization(colspanAttr), minColspan, maxColspan, defaultColspan);
}

unsigned HTMLTableCellElement::rowSpan() const
{
    // FIXME: a rowSpan equal to 0 should be allowed, and mean that the cell is to span all the remaining rows in the row group.
    return std::max(1u, rowSpanForBindings());
}

unsigned HTMLTableCellElement::rowSpanForBindings() const
{
    return clampHTMLNonNegativeIntegerToRange(attributeWithoutSynchronization(rowspanAttr), minRowspan, maxRowspan, defaultRowspan);
}

int HTMLTableCellElement::cellIndex() const
{
    int index = 0;
    if (!parentElement() || !parentElement()->hasTagName(trTag))
        return -1;

    for (const Node * node = previousSibling(); node; node = node->previousSibling()) {
        if (node->hasTagName(tdTag) || node->hasTagName(thTag))
            index++;
    }
    
    return index;
}

bool HTMLTableCellElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    switch (name.nodeName()) {
    case AttributeNames::nowrapAttr:
    case AttributeNames::widthAttr:
    case AttributeNames::heightAttr:
        return true;
    default:
        break;
    }
    return HTMLTablePartElement::hasPresentationalHintsForAttribute(name);
}

void HTMLTableCellElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    switch (name.nodeName()) {
    case AttributeNames::nowrapAttr:
        addPropertyToPresentationalHintStyle(style, CSSPropertyWhiteSpaceCollapse, CSSValueCollapse);
        addPropertyToPresentationalHintStyle(style, CSSPropertyTextWrapMode, CSSValueNowrap);
        break;
    case AttributeNames::widthAttr:
        // width="0" is not allowed for compatibility with WinIE.
        addHTMLLengthToStyle(style, CSSPropertyWidth, value, AllowZeroValue::No);
        break;
    case AttributeNames::heightAttr:
        // width="0" is not allowed for compatibility with WinIE.
        addHTMLLengthToStyle(style, CSSPropertyHeight, value, AllowZeroValue::No);
        break;
    default:
        HTMLTablePartElement::collectPresentationalHintsForAttribute(name, value, style);
        break;
    }
}

void HTMLTableCellElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    HTMLTablePartElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);

    if (name == rowspanAttr || name == colspanAttr) {
        if (CheckedPtr tableCell = dynamicDowncast<RenderTableCell>(renderer()))
            tableCell->colSpanOrRowSpanChanged();
    }
}

const MutableStyleProperties* HTMLTableCellElement::additionalPresentationalHintStyle() const
{
    if (auto table = findParentTable())
        return table->additionalCellStyle();
    return nullptr;
}

bool HTMLTableCellElement::isURLAttribute(const Attribute& attribute) const
{
    return attribute.name() == backgroundAttr || HTMLTablePartElement::isURLAttribute(attribute);
}

String HTMLTableCellElement::abbr() const
{
    return attributeWithoutSynchronization(abbrAttr);
}

String HTMLTableCellElement::axis() const
{
    return attributeWithoutSynchronization(axisAttr);
}

void HTMLTableCellElement::setColSpan(unsigned n)
{
    setAttributeWithoutSynchronization(colspanAttr, AtomString::number(limitToOnlyHTMLNonNegative(n, 1)));
}

String HTMLTableCellElement::headers() const
{
    return attributeWithoutSynchronization(headersAttr);
}

void HTMLTableCellElement::setRowSpanForBindings(unsigned n)
{
    setAttributeWithoutSynchronization(rowspanAttr, AtomString::number(limitToOnlyHTMLNonNegative(n, 1)));
}

const AtomString& HTMLTableCellElement::scope() const
{
    // https://html.spec.whatwg.org/multipage/tables.html#attr-th-scope
    static MainThreadNeverDestroyed<const AtomString> row("row"_s);
    static MainThreadNeverDestroyed<const AtomString> col("col"_s);
    static MainThreadNeverDestroyed<const AtomString> rowgroup("rowgroup"_s);
    static MainThreadNeverDestroyed<const AtomString> colgroup("colgroup"_s);

    const AtomString& value = attributeWithoutSynchronization(HTMLNames::scopeAttr);

    if (equalIgnoringASCIICase(value, row))
        return row;
    if (equalIgnoringASCIICase(value, col))
        return col;
    if (equalIgnoringASCIICase(value, rowgroup))
        return rowgroup;
    if (equalIgnoringASCIICase(value, colgroup))
        return colgroup;
    return emptyAtom();
}

void HTMLTableCellElement::setScope(const AtomString& scope)
{
    setAttributeWithoutSynchronization(scopeAttr, scope);
}

void HTMLTableCellElement::addSubresourceAttributeURLs(ListHashSet<URL>& urls) const
{
    HTMLTablePartElement::addSubresourceAttributeURLs(urls);

    addSubresourceURL(urls, document().completeURL(attributeWithoutSynchronization(backgroundAttr)));
}

HTMLTableCellElement* HTMLTableCellElement::cellAbove() const
{
    auto* tableCellRenderer = dynamicDowncast<RenderTableCell>(renderer());
    if (!tableCellRenderer)
        return nullptr;

    auto* cellAboveRenderer = tableCellRenderer->table()->cellAbove(tableCellRenderer);
    if (!cellAboveRenderer)
        return nullptr;

    return downcast<HTMLTableCellElement>(cellAboveRenderer->element());
}

} // namespace WebCore
