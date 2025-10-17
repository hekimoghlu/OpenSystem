/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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

#if USE(ATSPI)
#include "AXTextStateChangeIntent.h"
#include "AccessibilityObject.h"
#include "AccessibilityObjectAtspi.h"
#include "AccessibilityRenderObject.h"
#include "Document.h"
#include "Element.h"
#include "HTMLSelectElement.h"
#include "Range.h"
#include "TextIterator.h"

namespace WebCore {

void AXObjectCache::attachWrapper(AccessibilityObject& axObject)
{
    auto wrapper = AccessibilityObjectAtspi::create(&axObject, document()->page()->accessibilityRootObject());
    axObject.setWrapper(wrapper.ptr());

    m_deferredParentChangedList.add(&axObject);
    m_performCacheUpdateTimer.startOneShot(0_s);
}

void AXObjectCache::platformPerformDeferredCacheUpdate()
{
    auto handleParentChanged = [&](const AXCoreObject& axObject) {
        auto* wrapper = axObject.wrapper();
        if (!wrapper)
            return;

        auto* axParent = axObject.parentObjectUnignored();
        if (!axParent) {
            if (axObject.isScrollView() && document() && axObject.scrollView() == document()->view())
                wrapper->setParent(nullptr); // nullptr means root.
            return;
        }

        if (auto* axParentWrapper = axParent->wrapper())
            wrapper->setParent(axParentWrapper);
    };

    for (const auto& axObject : m_deferredParentChangedList)
        handleParentChanged(*axObject);
    m_deferredParentChangedList.clear();
}

void AXObjectCache::postPlatformNotification(AccessibilityObject& coreObject, AXNotification notification)
{
    auto* wrapper = coreObject.wrapper();
    if (!wrapper)
        return;

    switch (notification) {
    case AXNotification::CheckedStateChanged:
        if (coreObject.isCheckboxOrRadio() || coreObject.isSwitch())
            wrapper->stateChanged("checked", coreObject.isChecked());
        break;
    case AXNotification::SelectedStateChanged:
        wrapper->stateChanged("selected", coreObject.isSelected());
        break;
    case AXNotification::MenuListItemSelected: {
        // Menu list popup items are handled by AXSelectedStateChanged.
        auto* parent = coreObject.parentObjectUnignored();
        if (parent && !parent->isMenuListPopup())
            wrapper->stateChanged("selected", coreObject.isSelected());
        break;
    }
    case AXNotification::SelectedCellsChanged:
    case AXNotification::SelectedChildrenChanged:
        wrapper->selectionChanged();
        break;
    case AXNotification::MenuListValueChanged: {
        const auto& children = coreObject.children();
        if (children.size() == 1) {
            if (auto* childWrapper = children[0]->wrapper())
                childWrapper->selectionChanged();
        }
        break;
    }
    case AXNotification::ValueChanged:
        if (wrapper->interfaces().contains(AccessibilityObjectAtspi::Interface::Value))
            wrapper->valueChanged(coreObject.valueForRange());
        break;
    case AXNotification::InvalidStatusChanged:
        wrapper->stateChanged("invalid-entry", coreObject.invalidStatus() != "false"_s);
        break;
    case AXNotification::ElementBusyChanged:
        wrapper->stateChanged("busy", coreObject.isBusy());
        break;
    case AXNotification::CurrentStateChanged:
        wrapper->stateChanged("active", coreObject.currentState() != AccessibilityCurrentState::False);
        break;
    case AXNotification::RowExpanded:
        wrapper->stateChanged("expanded", true);
        break;
    case AXNotification::RowCollapsed:
        wrapper->stateChanged("expanded", false);
        break;
    case AXNotification::ExpandedChanged:
        wrapper->stateChanged("expanded", coreObject.isExpanded());
        break;
    case AXNotification::DisabledStateChanged: {
        bool enabledState = coreObject.isEnabled();
        wrapper->stateChanged("enabled", enabledState);
        wrapper->stateChanged("sensitive", enabledState);
        break;
    }
    case AXNotification::PressedStateChanged:
        wrapper->stateChanged("pressed", coreObject.isPressed());
        break;
    case AXNotification::ReadOnlyStatusChanged:
        wrapper->stateChanged("read-only", !coreObject.canSetValueAttribute());
        break;
    case AXNotification::RequiredStatusChanged:
        wrapper->stateChanged("required", coreObject.isRequired());
        break;
    case AXNotification::ActiveDescendantChanged:
        wrapper->activeDescendantChanged();
        break;
    case AXNotification::ChildrenChanged:
        coreObject.updateChildrenIfNecessary();
        break;
    default:
        break;
    }
}

void AXObjectCache::postTextStateChangePlatformNotification(AccessibilityObject* coreObject, const AXTextStateChangeIntent&, const VisibleSelection& selection)
{
    if (!coreObject)
        coreObject = rootWebArea();

    if (!coreObject)
        return;

    auto* wrapper = coreObject->wrapper();
    if (!wrapper)
        return;

    wrapper->selectionChanged(selection);
}

void AXObjectCache::postTextStateChangePlatformNotification(AccessibilityObject* coreObject, AXTextEditType editType, const String& text, const VisiblePosition& position)
{
    if (text.isEmpty())
        return;

    auto* wrapper = coreObject->wrapper();
    if (!wrapper)
        return;

    switch (editType) {
    case AXTextEditTypeDelete:
    case AXTextEditTypeCut:
        wrapper->textDeleted(text, position);
        break;
    case AXTextEditTypeInsert:
    case AXTextEditTypeTyping:
    case AXTextEditTypeDictation:
    case AXTextEditTypePaste:
        wrapper->textInserted(text, position);
        break;
    case AXTextEditTypeAttributesChange:
        wrapper->textAttributesChanged();
        break;
    case AXTextEditTypeUnknown:
        break;
    }
}

void AXObjectCache::postTextReplacementPlatformNotificationForTextControl(AccessibilityObject* coreObject, const String& deletedText, const String& insertedText)
{
    if (!coreObject)
        coreObject = rootWebArea();

    if (!coreObject)
        return;

    if (deletedText.isEmpty() && insertedText.isEmpty())
        return;

    auto* wrapper = coreObject->wrapper();
    if (!wrapper)
        return;

    if (!deletedText.isEmpty())
        wrapper->textDeleted(deletedText, coreObject->visiblePositionForIndex(0));
    if (!insertedText.isEmpty())
        wrapper->textInserted(insertedText, coreObject->visiblePositionForIndex(insertedText.length()));
}

void AXObjectCache::postTextReplacementPlatformNotification(AccessibilityObject* coreObject, AXTextEditType, const String& deletedText, AXTextEditType, const String& insertedText, const VisiblePosition& position)
{
    if (!coreObject)
        coreObject = rootWebArea();

    if (!coreObject)
        return;

    if (deletedText.isEmpty() && insertedText.isEmpty())
        return;

    auto* wrapper = coreObject->wrapper();
    if (!wrapper)
        return;

    if (!deletedText.isEmpty())
        wrapper->textDeleted(deletedText, position);
    if (!insertedText.isEmpty())
        wrapper->textInserted(insertedText, position);
}

void AXObjectCache::frameLoadingEventPlatformNotification(AccessibilityObject* coreObject, AXLoadingEvent loadingEvent)
{
    if (!coreObject)
        return;

    if (coreObject->roleValue() != AccessibilityRole::WebArea)
        return;

    auto* wrapper = coreObject->wrapper();
    if (!wrapper)
        return;

    switch (loadingEvent) {
    case AXLoadingEvent::Started:
        wrapper->stateChanged("busy", true);
        break;
    case AXLoadingEvent::Reloaded:
        wrapper->stateChanged("busy", true);
        wrapper->loadEvent("Reload");
        break;
    case AXLoadingEvent::Failed:
        wrapper->stateChanged("busy", false);
        wrapper->loadEvent("LoadStopped");
        break;
    case AXLoadingEvent::Finished:
        wrapper->stateChanged("busy", false);
        wrapper->loadEvent("LoadComplete");
        break;
    }
}

void AXObjectCache::platformHandleFocusedUIElementChanged(Element* oldFocus, Element* newFocus)
{
    if (auto* axObject = get(oldFocus)) {
        if (auto* wrapper = axObject->wrapper())
            wrapper->stateChanged("focused", false);
    }
    if (auto* axObject = getOrCreate(newFocus)) {
        if (auto* wrapper = axObject->wrapper())
            wrapper->stateChanged("focused", true);
    }
}

void AXObjectCache::handleScrolledToAnchor(const Node&)
{
}

} // namespace WebCore

#endif // USE(ATSPI)
