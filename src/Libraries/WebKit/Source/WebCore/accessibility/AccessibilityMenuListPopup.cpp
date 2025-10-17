/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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
#include "AccessibilityMenuListPopup.h"

#include "AXObjectCache.h"
#include "AccessibilityMenuList.h"
#include "AccessibilityMenuListOption.h"
#include "HTMLNames.h"
#include "HTMLOptionElement.h"
#include "HTMLSelectElement.h"
#include "RenderMenuList.h"

namespace WebCore {

using namespace HTMLNames;

AccessibilityMenuListPopup::AccessibilityMenuListPopup(AXID axID)
    : AccessibilityMockObject(axID)
{
}

bool AccessibilityMenuListPopup::isVisible() const
{
    return false;
}

bool AccessibilityMenuListPopup::isOffScreen() const
{
    if (!m_parent)
        return true;
    
    return m_parent->isCollapsed();
}

bool AccessibilityMenuListPopup::isEnabled() const
{
    if (!m_parent)
        return false;
    
    return m_parent->isEnabled();
}

bool AccessibilityMenuListPopup::computeIsIgnored() const
{
    return isIgnoredByDefault();
}

AccessibilityMenuListOption* AccessibilityMenuListPopup::menuListOptionAccessibilityObject(HTMLElement* element) const
{
    if (!element || !element->inRenderedDocument())
        return nullptr;

    return dynamicDowncast<AccessibilityMenuListOption>(document()->axObjectCache()->getOrCreate(*element));
}

bool AccessibilityMenuListPopup::press()
{
    if (!m_parent)
        return false;
    
    m_parent->press();
    return true;
}

void AccessibilityMenuListPopup::addChildren()
{
    if (!m_parent)
        return;

    RefPtr select = dynamicDowncast<HTMLSelectElement>(m_parent->node());
    if (!select)
        return;

    m_childrenInitialized = true;

    for (const auto& listItem : select->listItems()) {
        if (auto* menuListOptionObject = menuListOptionAccessibilityObject(listItem.get())) {
            menuListOptionObject->setParent(this);
            addChild(*menuListOptionObject, DescendIfIgnored::No);
        }
    }
}

void AccessibilityMenuListPopup::handleChildrenChanged()
{
    CheckedPtr cache = axObjectCache();
    if (!cache)
        return;

    const auto& children = unignoredChildren(/* updateChildrenIfNeeded */ false);
    for (size_t i = children.size(); i > 0; --i) {
        auto& child = children[i - 1].get();
        if (RefPtr actionElement = child.actionElement(); actionElement && !actionElement->inRenderedDocument()) {
            child.detachFromParent();
            cache->remove(child.objectID());
        }
    }

    m_children.clear();
    m_childrenInitialized = false;
    addChildren();
}

void AccessibilityMenuListPopup::didUpdateActiveOption(int optionIndex)
{
    ASSERT_ARG(optionIndex, optionIndex >= 0);
    const auto& children = unignoredChildren();
    ASSERT_ARG(optionIndex, optionIndex < static_cast<int>(children.size()));

    CheckedPtr cache = axObjectCache();
    if (!cache)
        return;

    auto& child = downcast<AccessibilityObject>(children[optionIndex].get());
    cache->postNotification(&child, document(), AXNotification::FocusedUIElementChanged);
    cache->postNotification(&child, document(), AXNotification::MenuListItemSelected);
}

} // namespace WebCore
