/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
#include "AccessibilityListBox.h"

#include "AXObjectCache.h"
#include "AccessibilityListBoxOption.h"
#include "HTMLNames.h"
#include "HTMLSelectElement.h"
#include "HitTestResult.h"
#include "RenderListBox.h"
#include "RenderObject.h"
#include <wtf/Scope.h>

namespace WebCore {

using namespace HTMLNames;

AccessibilityListBox::AccessibilityListBox(AXID axID, RenderObject& renderer)
    : AccessibilityRenderObject(axID, renderer)
{
}

AccessibilityListBox::~AccessibilityListBox() = default;

Ref<AccessibilityListBox> AccessibilityListBox::create(AXID axID, RenderObject& renderer)
{
    return adoptRef(*new AccessibilityListBox(axID, renderer));
}

void AccessibilityListBox::addChildren()
{
    m_childrenInitialized = true;
    auto clearDirtySubtree = makeScopeExit([&] {
        m_subtreeDirty = false;
    });

    auto* selectElement = dynamicDowncast<HTMLSelectElement>(node());
    if (!selectElement)
        return;

    for (const auto& listItem : selectElement->listItems())
        addChild(listBoxOptionAccessibilityObject(listItem.get()), DescendIfIgnored::No);
}

void AccessibilityListBox::setSelectedChildren(const AccessibilityChildrenVector& children)
{
    if (!canSetSelectedChildren())
        return;

    // Unselect any selected option.
    for (const auto& child : unignoredChildren()) {
        if (child->isSelected())
            child->setSelected(false);
    }

    for (const auto& object : children) {
        if (object->isListBoxOption())
            object->setSelected(true);
    }
}

AXCoreObject::AccessibilityChildrenVector AccessibilityListBox::visibleChildren()
{
    ASSERT(!m_renderer || is<RenderListBox>(m_renderer.get()));
    auto* listBox = dynamicDowncast<RenderListBox>(m_renderer.get());
    if (!listBox)
        return { };

    if (!childrenInitialized())
        addChildren();
    
    const auto& children = const_cast<AccessibilityListBox*>(this)->unignoredChildren();
    AccessibilityChildrenVector result;
    size_t size = children.size();
    for (size_t i = 0; i < size; i++) {
        if (listBox->listIndexIsVisible(i))
            result.append(children[i]);
    }
    return result;
}

AccessibilityObject* AccessibilityListBox::listBoxOptionAccessibilityObject(HTMLElement* element) const
{
    // FIXME: Why does AccessibilityMenuListPopup::menuListOptionAccessibilityObject check inRenderedDocument, but this does not?
    if (auto* document = this->document())
        return document->axObjectCache()->getOrCreate(element);
    return nullptr;
}

AccessibilityObject* AccessibilityListBox::elementAccessibilityHitTest(const IntPoint& point) const
{
    // the internal HTMLSelectElement methods for returning a listbox option at a point
    // ignore optgroup elements.
    if (!m_renderer)
        return nullptr;
    
    Node* node = m_renderer->node();
    if (!node)
        return nullptr;
    
    LayoutRect parentRect = boundingBoxRect();
    
    AccessibilityObject* listBoxOption = nullptr;
    const auto& children = const_cast<AccessibilityListBox*>(this)->unignoredChildren();
    unsigned length = children.size();
    for (unsigned i = 0; i < length; ++i) {
        LayoutRect rect = downcast<RenderListBox>(*m_renderer).itemBoundingBoxRect(parentRect.location(), i);
        // The cast to HTMLElement below is safe because the only other possible listItem type
        // would be a WMLElement, but WML builds don't use accessibility features at all.
        if (rect.contains(point)) {
            listBoxOption = &downcast<AccessibilityObject>(children[i].get());
            break;
        }
    }
    
    if (listBoxOption && !listBoxOption->isIgnored())
        return listBoxOption;
    
    return axObjectCache()->getOrCreate(renderer());
}

} // namespace WebCore
