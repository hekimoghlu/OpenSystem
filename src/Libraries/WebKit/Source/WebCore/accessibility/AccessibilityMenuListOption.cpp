/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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
#include "AccessibilityMenuListOption.h"

#include "AXObjectCache.h"
#include "AccessibilityMenuListPopup.h"
#include "Document.h"
#include "HTMLNames.h"
#include "HTMLOptionElement.h"
#include "HTMLSelectElement.h"

namespace WebCore {

using namespace HTMLNames;

AccessibilityMenuListOption::AccessibilityMenuListOption(AXID axID, HTMLOptionElement& element)
    : AccessibilityNodeObject(axID, &element)
    , m_parent(nullptr)
{
}

Ref<AccessibilityMenuListOption> AccessibilityMenuListOption::create(AXID axID, HTMLOptionElement& element)
{
    return adoptRef(*new AccessibilityMenuListOption(axID, element));
}

HTMLOptionElement* AccessibilityMenuListOption::optionElement() const
{
    return downcast<HTMLOptionElement>(node());
}

Element* AccessibilityMenuListOption::actionElement() const
{
    return downcast<Element>(node());
}

bool AccessibilityMenuListOption::isEnabled() const
{
    auto* optionElement = this->optionElement();
    return optionElement && !optionElement->ownElementDisabled();
}

bool AccessibilityMenuListOption::isVisible() const
{
    WeakPtr optionElement = this->optionElement();
    if (!optionElement)
        return false;

    // In a single-option select with the popup collapsed, only the selected item is considered visible.
    auto* ownerSelectElement = optionElement->document().axObjectCache()->getOrCreate(optionElement->ownerSelectElement());
    return ownerSelectElement && (!ownerSelectElement->isOffScreen() || isSelected());
}

bool AccessibilityMenuListOption::isOffScreen() const
{
    // Invisible list options are considered to be offscreen.
    return !isVisible();
}

bool AccessibilityMenuListOption::isSelected() const
{
    auto* optionElement = this->optionElement();
    return optionElement && optionElement->selected();
}

void AccessibilityMenuListOption::setSelected(bool selected)
{
    if (!canSetSelectedAttribute())
        return;
    
    if (auto* optionElement = this->optionElement())
        optionElement->setSelected(selected);
}

bool AccessibilityMenuListOption::canSetSelectedAttribute() const
{
    return isEnabled();
}

bool AccessibilityMenuListOption::computeIsIgnored() const
{
    return isIgnoredByDefault();
}

LayoutRect AccessibilityMenuListOption::elementRect() const
{
    RefPtr parent = parentObject();
    // Our parent should've been set to be a menu-list popup before this method is called.
    ASSERT(parent && parent->isMenuListPopup());
    if (!parent)
        return boundingBoxRect();

    RefPtr grandparent = parent->parentObject();
    ASSERT(!grandparent || grandparent->isMenuList());

    return grandparent ? grandparent->elementRect() : boundingBoxRect();
}

String AccessibilityMenuListOption::stringValue() const
{
    auto* optionElement = this->optionElement();
    return optionElement ? optionElement->label() : String();
}

} // namespace WebCore
