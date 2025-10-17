/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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
#include "AXObjectCache.h"

#include "AccessibilityObject.h"
#include "Chrome.h"
#include "ChromeClient.h"
#include "HTMLSelectElement.h"
#include "LocalFrame.h"
#include "Page.h"

namespace WebCore {

void AXObjectCache::attachWrapper(AccessibilityObject& object)
{
    auto wrapper = adoptRef(*new AccessibilityObjectWrapper());
    object.setWrapper(wrapper.ptr());
}

void AXObjectCache::detachWrapper(AXCoreObject*, AccessibilityDetachmentType)
{
}

static AXCoreObject* notifyChildrenSelectionChange(AXCoreObject* object)
{
    // Only list boxes supported so far.
    if (!object || !object->isListBox())
        return object;

    // Only support HTML select elements so far (ARIA selectors not supported).
    Node* node = object->node();
    if (!is<HTMLSelectElement>(node))
        return object;

    // Find the item where the selection change was triggered from.
    HTMLSelectElement& select = downcast<HTMLSelectElement>(*node);
    int changedItemIndex = select.activeSelectionStartListIndex();

    const AccessibilityObject::AccessibilityChildrenVector& items = object->children();
    if (changedItemIndex < 0 || changedItemIndex >= static_cast<int>(items.size()))
        return object;
    return items.at(changedItemIndex).ptr();
}

static AXNotification checkInteractableObjects(AXCoreObject* object)
{
    if (!object->isEnabled())
        return AXNotification::PressDidFail;

    if (object->isTextControl() && !object->canSetValueAttribute()) // Also determine whether it is readonly
        return AXNotification::PressDidFail;

    return AXNotification::PressDidSucceed;
}

void AXObjectCache::postPlatformNotification(AccessibilityObject& object, AXNotification notification)
{
    if (!document()
        || !!object.document()
        || !object.document()->view()
        || object.document()->view()->layoutContext().layoutState()
        || object.document()->childNeedsStyleRecalc())
        return;

    RefPtr protectedObject = &object;
    switch (notification) {
    case AXNotification::SelectedChildrenChanged:
        protectedObject = downcast<AccessibilityObject>(notifyChildrenSelectionChange(protectedObject.get()));
        break;
    case AXNotification::PressDidSucceed:
        notification = checkInteractableObjects(protectedObject.get());
        break;
    default:
        break;
    }

    ChromeClient& client = document()->frame()->page()->chrome().client();
    client.postAccessibilityNotification(*protectedObject, notification);
}

void AXObjectCache::nodeTextChangePlatformNotification(AccessibilityObject* object, AXTextChange textChange, unsigned offset, const String& text)
{
    if (!document()
        || !object
        || !object->document()
        || !object->document()->view()
        || object->document()->view()->layoutContext().layoutState()
        || object->document()->childNeedsStyleRecalc())
        return;
    ChromeClient& client = document()->frame()->page()->chrome().client();
    client.postAccessibilityNodeTextChangeNotification(object, textChange, offset, text);
}

void AXObjectCache::frameLoadingEventPlatformNotification(AccessibilityObject* object, AXLoadingEvent loadingEvent)
{
    if (!document()
        || !object
        || !object->document()
        || !object->document()->view()
        || object->document()->view()->layoutContext().layoutState()
        || object->document()->childNeedsStyleRecalc())
        return;
    ChromeClient& client = document()->frame()->page()->chrome().client();
    client.postAccessibilityFrameLoadingEventNotification(object, loadingEvent);
}

void AXObjectCache::handleScrolledToAnchor(const Node& scrolledToNode)
{
    if (RefPtr object = AccessibilityObject::firstAccessibleObjectFromNode(&scrolledToNode))
        postPlatformNotification(*object, AXNotification::ScrolledToAnchor);
}

void AXObjectCache::platformHandleFocusedUIElementChanged(Element*, Element* newFocus)
{
    if (!newFocus)
        return;

    Page* page = newFocus->document().page();
    if (!page || !page->chrome().platformPageClient())
        return;

    if (RefPtr focusedObject = focusedObjectForPage(page))
        postPlatformNotification(*focusedObject, AXNotification::FocusedUIElementChanged);
}

void AXObjectCache::platformPerformDeferredCacheUpdate()
{
}

} // namespace WebCore
