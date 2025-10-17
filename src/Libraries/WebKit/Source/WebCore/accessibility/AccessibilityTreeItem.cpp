/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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
#include "AccessibilityTreeItem.h"

#include "AXObjectCache.h"
#include "HTMLNames.h"

namespace WebCore {
    
using namespace HTMLNames;
    
AccessibilityTreeItem::AccessibilityTreeItem(AXID axID, RenderObject& renderer)
    : AccessibilityRenderObject(axID, renderer)
{
}

AccessibilityTreeItem::AccessibilityTreeItem(AXID axID, Node& node)
    : AccessibilityRenderObject(axID, node)
{
}

AccessibilityTreeItem::~AccessibilityTreeItem() = default;
    
Ref<AccessibilityTreeItem> AccessibilityTreeItem::create(AXID axID, RenderObject& renderer)
{
    return adoptRef(*new AccessibilityTreeItem(axID, renderer));
}

Ref<AccessibilityTreeItem> AccessibilityTreeItem::create(AXID axID, Node& node)
{
    return adoptRef(*new AccessibilityTreeItem(axID, node));
}

bool AccessibilityTreeItem::supportsCheckedState() const
{
    return hasAttribute(aria_checkedAttr);
}

AccessibilityRole AccessibilityTreeItem::determineAccessibilityRole()
{
    // Walk the parent chain looking for a parent that is a tree. A treeitem is
    // only considered valid if it is in a tree.
    AccessibilityObject* parent = nullptr;
    for (parent = parentObject(); parent && !parent->isTree(); parent = parent->parentObject()) { }
    m_isTreeItemValid = parent;

    return AccessibilityRenderObject::determineAccessibilityRole();
}
    
} // namespace WebCore
