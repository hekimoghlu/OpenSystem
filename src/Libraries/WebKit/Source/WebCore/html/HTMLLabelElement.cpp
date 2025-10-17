/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 18, 2025.
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
#include "HTMLLabelElement.h"

#include "Document.h"
#include "Event.h"
#include "EventNames.h"
#include "FormListedElement.h"
#include "HTMLFormControlElement.h"
#include "HTMLNames.h"
#include "MouseEvent.h"
#include "SelectionRestorationMode.h"
#include "TypedElementDescendantIteratorInlines.h"
#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLLabelElement);

using namespace HTMLNames;

static HTMLElement* firstElementWithIdIfLabelable(TreeScope& treeScope, const AtomString& id)
{
    if (RefPtr element = treeScope.getElementById(id)) {
        if (auto* labelableElement = dynamicDowncast<HTMLElement>(*element)) {
            if (labelableElement->isLabelable())
                return labelableElement;
        }
    }
    return nullptr;
}

inline HTMLLabelElement::HTMLLabelElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(labelTag));
}

Ref<HTMLLabelElement> HTMLLabelElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLLabelElement(tagName, document));
}

Ref<HTMLLabelElement> HTMLLabelElement::create(Document& document)
{
    return adoptRef(*new HTMLLabelElement(labelTag, document));
}

RefPtr<HTMLElement> HTMLLabelElement::control() const
{
    auto& controlId = attributeWithoutSynchronization(forAttr);
    if (controlId.isNull()) {
        // Search the children and descendants of the label element for a form element.
        // per http://dev.w3.org/html5/spec/Overview.html#the-label-element
        // the form element must be "labelable form-associated element".
        for (const auto& labelableElement : descendantsOfType<HTMLElement>(*this)) {
            if (labelableElement.isLabelable())
                return const_cast<HTMLElement*>(&labelableElement);
        }
        return nullptr;
    }
    return isConnected() ? firstElementWithIdIfLabelable(treeScope(), controlId) : nullptr;
}

HTMLFormElement* HTMLLabelElement::form() const
{
    if (auto element = control()) {
        if (auto* listedElement = element->asValidatedFormListedElement())
            return listedElement->form();
    }
    return nullptr;
}

void HTMLLabelElement::setActive(bool down, Style::InvalidationScope invalidationScope)
{
    if (down == active())
        return;

    // Update our status first.
    HTMLElement::setActive(down, invalidationScope);

    // Also update our corresponding control.
    if (auto element = control())
        element->setActive(down);
}

void HTMLLabelElement::setHovered(bool over, Style::InvalidationScope invalidationScope, HitTestRequest request)
{
    if (over == hovered())
        return;
        
    // Update our status first.
    HTMLElement::setHovered(over, invalidationScope, request);

    // Also update our corresponding control.
    if (auto element = control())
        element->setHovered(over);
}

bool HTMLLabelElement::isEventTargetedAtInteractiveDescendants(Event& event) const
{
    RefPtr node = dynamicDowncast<Node>(*event.target());
    if (!node)
        return false;

    if (!containsIncludingShadowDOM(node.get()))
        return false;

    for (const auto* it = node.get(); it && it != this; it = it->parentElementInComposedTree()) {
        auto* element = dynamicDowncast<HTMLElement>(*it);
        if (element && element->isInteractiveContent())
            return true;
    }

    return false;
}
void HTMLLabelElement::defaultEventHandler(Event& event)
{
    if (isAnyClick(event) && !m_processingClick) {
        auto control = this->control();

        // If we can't find a control or if the control received the click
        // event, then there's no need for us to do anything.
        auto* eventTarget = dynamicDowncast<Node>(event.target());
        if (!control || (eventTarget && control->containsIncludingShadowDOM(eventTarget))) {
            HTMLElement::defaultEventHandler(event);
            return;
        }

        // The activation behavior of a label element for events targeted at interactive
        // content descendants of a label element, and any descendants of those interactive
        // content descendants, must be to do nothing.
        // https://html.spec.whatwg.org/#the-label-element
        if (isEventTargetedAtInteractiveDescendants(event)) {
            HTMLElement::defaultEventHandler(event);
            return;
        }

        SetForScope processingClick(m_processingClick, true);

        control->dispatchSimulatedClick(&event);

        protectedDocument()->updateLayoutIgnorePendingStylesheets();
        if (control->isMouseFocusable())
            control->focus({ { }, { }, { }, FocusTrigger::Click, { } });

        event.setDefaultHandled();
    }

    HTMLElement::defaultEventHandler(event);
}

bool HTMLLabelElement::willRespondToMouseClickEventsWithEditability(Editability editability) const
{
    auto element = control();
    return (element && element->willRespondToMouseClickEventsWithEditability(editability)) || HTMLElement::willRespondToMouseClickEventsWithEditability(editability);
}

void HTMLLabelElement::focus(const FocusOptions& options)
{
    Ref<HTMLLabelElement> protectedThis(*this);
    auto document = protectedDocument();
    if (document->haveStylesheetsLoaded()) {
        document->updateLayout();
        if (isFocusable()) {
            // The value of restorationMode is not used for label elements as it doesn't override updateFocusAppearance.
            Element::focus(options);
            return;
        }
    }

    // To match other browsers, always restore previous selection.
    if (auto element = control())
        element->focus({ SelectionRestorationMode::RestoreOrSelectAll, options.direction });
}

bool HTMLLabelElement::accessKeyAction(bool sendMouseEvents)
{
    if (auto element = control())
        return element->accessKeyAction(sendMouseEvents);

    return HTMLElement::accessKeyAction(sendMouseEvents);
}

auto HTMLLabelElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree) -> InsertedIntoAncestorResult
{
    auto result = HTMLElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);

    if (parentOfInsertedTree.isInTreeScope() && insertionType.treeScopeChanged) {
        auto& newScope = parentOfInsertedTree.treeScope();
        if (newScope.shouldCacheLabelsByForAttribute())
            updateLabel(newScope, nullAtom(), attributeWithoutSynchronization(forAttr));
    }

    return result;
}

void HTMLLabelElement::updateLabel(TreeScope& scope, const AtomString& oldForAttributeValue, const AtomString& newForAttributeValue)
{
    if (!isConnected())
        return;

    if (oldForAttributeValue == newForAttributeValue)
        return;

    if (!oldForAttributeValue.isEmpty())
        scope.removeLabel(oldForAttributeValue, *this);
    if (!newForAttributeValue.isEmpty())
        scope.addLabel(newForAttributeValue, *this);
}

void HTMLLabelElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    if (oldParentOfRemovedTree.isInTreeScope() && removalType.treeScopeChanged) {
        auto& oldScope = oldParentOfRemovedTree.treeScope();
        if (oldScope.shouldCacheLabelsByForAttribute())
            updateLabel(oldScope, attributeWithoutSynchronization(forAttr), nullAtom());
    }

    HTMLElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
}

} // namespace
