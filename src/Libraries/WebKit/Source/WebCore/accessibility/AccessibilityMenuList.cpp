/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#include "AccessibilityMenuList.h"

#include "AXObjectCache.h"
#include "AccessibilityMenuListPopup.h"
#include "RenderMenuList.h"
#include <wtf/Scope.h>

namespace WebCore {

AccessibilityMenuList::AccessibilityMenuList(AXID axID, RenderMenuList& renderer, AXObjectCache& cache)
    : AccessibilityRenderObject(axID, renderer)
    , m_popup(downcast<AccessibilityMenuListPopup>(*cache.create(AccessibilityRole::MenuListPopup)))
{
    m_popup->setParent(this);

    addChild(m_popup.get());
    m_childrenInitialized = true;
}

Ref<AccessibilityMenuList> AccessibilityMenuList::create(AXID axID, RenderMenuList& renderer, AXObjectCache& cache)
{
    return adoptRef(*new AccessibilityMenuList(axID, renderer, cache));
}

bool AccessibilityMenuList::press()
{
    if (!m_renderer)
        return false;

#if !PLATFORM(IOS_FAMILY)
    RefPtr element = this->element();
    auto notification = AXNotification::PressDidFail;
    if (CheckedPtr menuList = dynamicDowncast<RenderMenuList>(renderer()); menuList && element && !element->isDisabledFormControl()) {
        if (menuList->popupIsVisible())
            menuList->hidePopup();
        else
            menuList->showPopup();
        notification = AXNotification::PressDidSucceed;
    }
    if (CheckedPtr cache = axObjectCache())
        cache->postNotification(element.get(), notification);
    return true;
#endif
    return false;
}

void AccessibilityMenuList::updateChildrenIfNecessary()
{
    // Typically for AccessibilityNodeObject subclasses, updateChildrenIfNecessary() is what
    // calls addChildren(), which in turn passes m_subtreeDirty down the tree as objects are inserted.
    // However, we purposely never allow our children to be cleared or become unitialized, which
    // by the definition of the AccessibilityNodeObject::updateChildrenIfNecessary() means addChildren()
    // will never be called. (We add our only child, m_popup, once in the constructor).
    //
    // Despite this, we still want to pass down the m_subtreeDirty flag if we have it set, so do that here.
    if (m_subtreeDirty)
        m_popup->setNeedsToUpdateSubtree();

    m_subtreeDirty = false;
}

void AccessibilityMenuList::addChildren()
{
    // This class sets its children once in the constructor, and should never
    // have dirty or uninitialized children afterwards.
    ASSERT(m_childrenInitialized);
    ASSERT(!m_childrenDirty);
}

bool AccessibilityMenuList::isCollapsed() const
{
    // Collapsed is the "default" state, so if the renderer doesn't exist
    // this makes slightly more sense than returning false.
    if (!m_renderer)
        return true;

#if !PLATFORM(IOS_FAMILY)
    CheckedPtr menuList = dynamicDowncast<RenderMenuList>(renderer());
    return !(menuList && menuList->popupIsVisible());
#else
    return true;
#endif
}

bool AccessibilityMenuList::canSetFocusAttribute() const
{
    RefPtr element = this->element();
    return element && !element->isDisabledFormControl();
}

void AccessibilityMenuList::didUpdateActiveOption(int optionIndex)
{
    RefPtr document = m_renderer ? &m_renderer->document() : nullptr;
    if (!document)
        return;

    const auto& childObjects = unignoredChildren();
    if (!childObjects.isEmpty()) {
        ASSERT(childObjects.size() == 1);
        ASSERT(is<AccessibilityMenuListPopup>(childObjects[0].get()));

        // We might be calling this method in situations where the renderers for list items
        // associated to the menu list have not been created (e.g. they might be rendered
        // in the UI process, as it's the case in the GTK+ port, which uses GtkMenuItem).
        // So, we need to make sure that the accessibility popup object has some children
        // before asking it to update its active option, or it will read invalid memory.
        // You can reproduce the issue in the GTK+ port by removing this check and running
        // accessibility/insert-selected-option-into-select-causes-crash.html (will crash).
        int popupChildrenSize = static_cast<int>(childObjects[0]->unignoredChildren().size());
        RefPtr accessibilityMenuListPopup = dynamicDowncast<AccessibilityMenuListPopup>(childObjects[0].get());
        if (accessibilityMenuListPopup && optionIndex >= 0 && optionIndex < popupChildrenSize)
            accessibilityMenuListPopup->didUpdateActiveOption(optionIndex);
    }

    if (CheckedPtr cache = document->axObjectCache())
        cache->deferMenuListValueChange(element());
}

} // namespace WebCore
