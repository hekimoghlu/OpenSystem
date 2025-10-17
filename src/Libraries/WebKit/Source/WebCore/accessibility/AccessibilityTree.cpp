/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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
#include "AccessibilityTree.h"

#include "AXObjectCache.h"
#include "AccessibilityTreeItem.h"
#include "Element.h"
#include "HTMLNames.h"

#include <wtf/Deque.h>

namespace WebCore {

using namespace HTMLNames;
    
AccessibilityTree::AccessibilityTree(AXID axID, RenderObject& renderer)
    : AccessibilityRenderObject(axID, renderer)
{
}

AccessibilityTree::AccessibilityTree(AXID axID, Node& node)
    : AccessibilityRenderObject(axID, node)
{
}

AccessibilityTree::~AccessibilityTree() = default;
    
Ref<AccessibilityTree> AccessibilityTree::create(AXID axID, RenderObject& renderer)
{
    return adoptRef(*new AccessibilityTree(axID, renderer));
}

Ref<AccessibilityTree> AccessibilityTree::create(AXID axID, Node& node)
{
    return adoptRef(*new AccessibilityTree(axID, node));
}

bool AccessibilityTree::computeIsIgnored() const
{
    return isIgnoredByDefault();
}

AccessibilityRole AccessibilityTree::determineAccessibilityRole()
{
    if ((m_ariaRole = determineAriaRoleAttribute()) != AccessibilityRole::Tree)
        return AccessibilityRenderObject::determineAccessibilityRole();

    return isTreeValid() ? AccessibilityRole::Tree : AccessibilityRole::Generic;
}

bool AccessibilityTree::isTreeValid() const
{
    // A valid tree can only have treeitem or group of treeitems as a child.
    // https://www.w3.org/TR/wai-aria/#tree
    Node* node = this->node();
    if (!node)
        return false;
    
    Deque<Ref<Node>> queue;
    for (RefPtr child = node->firstChild(); child; child = queue.last()->nextSibling())
        queue.append(child.releaseNonNull());

    while (!queue.isEmpty()) {
        Ref child = queue.takeFirst();

        auto* childElement = dynamicDowncast<Element>(child.get());
        if (!childElement)
            continue;
        if (hasRole(*childElement, "treeitem"_s))
            continue;
        if (!hasAnyRole(*childElement, { "group"_s, "presentation"_s }))
            return false;

        for (RefPtr groupChild = child->firstChild(); groupChild; groupChild = queue.last()->nextSibling())
            queue.append(groupChild.releaseNonNull());
    }
    return true;
}

} // namespace WebCore
