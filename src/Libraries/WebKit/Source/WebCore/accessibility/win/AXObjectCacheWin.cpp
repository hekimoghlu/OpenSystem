/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#include "Document.h"
#include "Page.h"
#include "RenderObject.h"

// Provided by IAccessibleEventID.idl
#define IA2_EVENT_DOCUMENT_LOAD_COMPLETE    261


namespace WebCore {

void AXObjectCache::detachWrapper(AXCoreObject* obj, AccessibilityDetachmentType)
{
    // On Windows, AccessibilityObjects are created when get_accChildCount is
    // called, but they are not wrapped until get_accChild is called, so this
    // object may not have a wrapper.
    if (AccessibilityObjectWrapper* wrapper = obj->wrapper())
        wrapper->detach();
}

void AXObjectCache::attachWrapper(AccessibilityObject&)
{
    // On Windows, AccessibilityObjects are wrapped when the accessibility
    // software requests them via get_accChild.
}

void AXObjectCache::handleScrolledToAnchor(const Node& anchorNode)
{
    // The anchor node may not be accessible. Post the notification for the
    // first accessible object.
    if (RefPtr object = AccessibilityObject::firstAccessibleObjectFromNode(&anchorNode))
        postPlatformNotification(*object, AXNotification::ScrolledToAnchor);
}

void AXObjectCache::postPlatformNotification(AccessibilityObject& object, AXNotification notification)
{
    Document* document = object.document();
    if (!document)
        return;

    Page* page = document->page();
    if (!page || !page->chrome().platformPageClient())
        return;

    DWORD msaaEvent;
    switch (notification) {
    case AXNotification::CheckedStateChanged:
        msaaEvent = EVENT_OBJECT_STATECHANGE;
        break;

    case AXNotification::FocusedUIElementChanged:
    case AXNotification::ActiveDescendantChanged:
        msaaEvent = EVENT_OBJECT_FOCUS;
        break;

    case AXNotification::ScrolledToAnchor:
        msaaEvent = EVENT_SYSTEM_SCROLLINGSTART;
        break;

    case AXNotification::LayoutComplete:
        msaaEvent = EVENT_OBJECT_REORDER;
        break;

    case AXNotification::LoadComplete:
        msaaEvent = IA2_EVENT_DOCUMENT_LOAD_COMPLETE;
        break;

    case AXNotification::ValueChanged:
    case AXNotification::MenuListValueChanged:
        msaaEvent = EVENT_OBJECT_VALUECHANGE;
        break;

    case AXNotification::MenuListItemSelected:
        msaaEvent = EVENT_OBJECT_SELECTION;
        break;

    default:
        return;
    }

    // Windows will end up calling get_accChild() on the root accessible
    // object for the WebView, passing the child ID that we specify below. We
    // negate the AXID so we know that the caller is passing the ID of an
    // element, not the index of a child element.

    ASSERT(object.objectID().toUInt64() >= 1);
    ASSERT(object.objectID().toUInt64() <= std::numeric_limits<LONG>::max());

    auto objectID = object.objectID();
    NotifyWinEvent(msaaEvent, page->chrome().platformPageClient(), OBJID_CLIENT, -static_cast<LONG>(objectID.toUInt64()));
}

void AXObjectCache::nodeTextChangePlatformNotification(AccessibilityObject*, AXTextChange, unsigned, const String&)
{
}

void AXObjectCache::frameLoadingEventPlatformNotification(AccessibilityObject* obj, AXLoadingEvent notification)
{
    if (!obj)
        return;
    
    Document* document = obj->document();
    if (!document)
        return;

    Page* page = document->page();
    if (!page)
        return;

    if (notification == AXLoadingEvent::Started)
        page->chrome().client().AXStartFrameLoad();
    else if (notification == AXLoadingEvent::Finished)
        page->chrome().client().AXFinishFrameLoad();
}

void AXObjectCache::platformHandleFocusedUIElementChanged(Element*, Element* newFocus)
{
    if (!newFocus)
        return;

    Page* page = newFocus->document().page();
    if (!page || !page->chrome().platformPageClient())
        return;

    if (RefPtr focusedObject = focusedObjectForPage(page)) {
        ASSERT(!focusedObject->isIgnored());
        postPlatformNotification(*focusedObject, AXNotification::FocusedUIElementChanged);
    }
}

void AXObjectCache::platformPerformDeferredCacheUpdate()
{
}

} // namespace WebCore
