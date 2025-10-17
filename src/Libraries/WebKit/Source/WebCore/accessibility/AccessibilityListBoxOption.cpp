/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
#include "AccessibilityListBoxOption.h"

#include "AXObjectCache.h"
#include "AccessibilityListBox.h"
#include "ElementInlines.h"
#include "HTMLNames.h"
#include "HTMLOptGroupElement.h"
#include "HTMLOptionElement.h"
#include "HTMLSelectElement.h"
#include "IntRect.h"
#include "RenderListBox.h"

namespace WebCore {

using namespace HTMLNames;

AccessibilityListBoxOption::AccessibilityListBoxOption(AXID axID, HTMLElement& element)
    : AccessibilityNodeObject(axID, &element)
{
}

AccessibilityListBoxOption::~AccessibilityListBoxOption() = default;

Ref<AccessibilityListBoxOption> AccessibilityListBoxOption::create(AXID axID, HTMLElement& element)
{
    return adoptRef(*new AccessibilityListBoxOption(axID, element));
}

bool AccessibilityListBoxOption::isEnabled() const
{
    return !(is<HTMLOptGroupElement>(m_node.get())
        || equalLettersIgnoringASCIICase(getAttribute(aria_disabledAttr), "true"_s)
        || hasAttribute(disabledAttr));
}

bool AccessibilityListBoxOption::isSelected() const
{
    RefPtr option = dynamicDowncast<HTMLOptionElement>(m_node.get());
    return option && option->selected();
}

bool AccessibilityListBoxOption::isSelectedOptionActive() const
{
    HTMLSelectElement* listBoxParentNode = listBoxOptionParentNode();
    if (!listBoxParentNode)
        return false;

    return listBoxParentNode->activeSelectionEndListIndex() == listBoxOptionIndex();
}

LayoutRect AccessibilityListBoxOption::elementRect() const
{
    if (!m_node)
        return { };

    RefPtr listBoxParentNode = listBoxOptionParentNode();
    if (!listBoxParentNode)
        return { };

    auto* listBoxRenderer = dynamicDowncast<RenderListBox>(listBoxParentNode->renderer());
    if (!listBoxRenderer)
        return { };

    WeakPtr cache = listBoxRenderer->document().axObjectCache();
    RefPtr listbox = cache ? cache->getOrCreate(*listBoxRenderer) : nullptr;
    if (!listbox)
        return { };

    auto parentRect = listbox->boundingBoxRect();
    int index = listBoxOptionIndex();
    if (index != -1)
        return listBoxRenderer->itemBoundingBoxRect(parentRect.location(), index);
    return { };
}

bool AccessibilityListBoxOption::computeIsIgnored() const
{
    if (!m_node || isIgnoredByDefault())
        return true;

    auto* parent = parentObject();
    return parent ? parent->isIgnored() : true;
}

bool AccessibilityListBoxOption::canSetSelectedAttribute() const
{
    RefPtr optionElement = dynamicDowncast<HTMLOptionElement>(m_node.get());
    if (!optionElement)
        return false;

    if (optionElement->isDisabledFormControl())
        return false;

    RefPtr selectElement = listBoxOptionParentNode();
    return !selectElement || !selectElement->isDisabledFormControl();
}

String AccessibilityListBoxOption::stringValue() const
{
    if (!m_node)
        return { };

    auto ariaLabel = getAttributeTrimmed(aria_labelAttr);
    if (!ariaLabel.isEmpty())
        return ariaLabel;

    if (RefPtr option = dynamicDowncast<HTMLOptionElement>(*m_node))
        return option->label();

    if (RefPtr optgroup = dynamicDowncast<HTMLOptGroupElement>(*m_node))
        return optgroup->groupLabelText();

    return { };
}

Element* AccessibilityListBoxOption::actionElement() const
{
    ASSERT(is<HTMLElement>(m_node.get()));
    return dynamicDowncast<Element>(m_node.get());
}

AccessibilityObject* AccessibilityListBoxOption::parentObject() const
{
    auto* parentNode = listBoxOptionParentNode();
    if (!parentNode)
        return nullptr;

    auto* cache = m_node->document().axObjectCache();
    return cache ? cache->getOrCreate(*parentNode) : nullptr;
}

void AccessibilityListBoxOption::setSelected(bool selected)
{
    HTMLSelectElement* selectElement = listBoxOptionParentNode();
    if (!selectElement)
        return;
    
    if (!canSetSelectedAttribute())
        return;
    
    bool isOptionSelected = isSelected();
    if ((isOptionSelected && selected) || (!isOptionSelected && !selected))
        return;
    
    // Convert from the entire list index to the option index.
    int optionIndex = selectElement->listToOptionIndex(listBoxOptionIndex());
    selectElement->accessKeySetSelectedIndex(optionIndex);
}

HTMLSelectElement* AccessibilityListBoxOption::listBoxOptionParentNode() const
{
    if (!m_node)
        return nullptr;

    if (RefPtr option = dynamicDowncast<HTMLOptionElement>(*m_node))
        return option->ownerSelectElement();

    if (RefPtr optgroup = dynamicDowncast<HTMLOptGroupElement>(*m_node))
        return optgroup->ownerSelectElement();

    return nullptr;
}

int AccessibilityListBoxOption::listBoxOptionIndex() const
{
    if (!m_node)
        return -1;

    auto* selectElement = listBoxOptionParentNode();
    if (!selectElement)
        return -1;

    const auto& listItems = selectElement->listItems();
    unsigned length = listItems.size();
    for (unsigned i = 0; i < length; i++) {
        if (listItems[i] == m_node)
            return i;
    }

    return -1;
}

} // namespace WebCore
