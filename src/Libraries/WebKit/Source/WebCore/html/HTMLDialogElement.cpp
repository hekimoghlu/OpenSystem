/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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
#include "HTMLDialogElement.h"

#include "CSSSelector.h"
#include "DocumentInlines.h"
#include "EventLoop.h"
#include "EventNames.h"
#include "FocusOptions.h"
#include "HTMLElement.h"
#include "HTMLNames.h"
#include "Logging.h"
#include "PopoverData.h"
#include "PseudoClassChangeInvalidation.h"
#include "RenderBlock.h"
#include "RenderElement.h"
#include "ScopedEventQueue.h"
#include "TypedElementDescendantIteratorInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLDialogElement);

using namespace HTMLNames;

HTMLDialogElement::HTMLDialogElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
}

ExceptionOr<void> HTMLDialogElement::show()
{
    // If the element already has an open attribute, then return.
    if (isOpen()) {
        if (!isModal())
            return { };
        return Exception { ExceptionCode::InvalidStateError, "Cannot call show() on an open modal dialog."_s };
    }

    setBooleanAttribute(openAttr, true);

    m_previouslyFocusedElement = document().focusedElement();

    auto hideUntil = topmostPopoverAncestor(TopLayerElementType::Other);

    document().hideAllPopoversUntil(hideUntil, FocusPreviousElement::No, FireEvents::No);

    runFocusingSteps();
    return { };
}

ExceptionOr<void> HTMLDialogElement::showModal()
{
    // If subject already has an open attribute, then throw an "InvalidStateError" DOMException.
    if (isOpen()) {
        if (isModal())
            return { };
        return Exception { ExceptionCode::InvalidStateError, "Cannot call showModal() on an open non-modal dialog."_s };
    }

    // If subject is not connected, then throw an "InvalidStateError" DOMException.
    if (!isConnected())
        return Exception { ExceptionCode::InvalidStateError, "Element is not connected."_s };

    if (isPopoverShowing())
        return Exception { ExceptionCode::InvalidStateError, "Element is already an open popover."_s };

    if (!protectedDocument()->isFullyActive())
        return Exception { ExceptionCode::InvalidStateError, "Invalid for dialogs within documents that are not fully active."_s };

    // setBooleanAttribute will dispatch a DOMSubtreeModified event.
    // Postpone callback execution that can potentially make the dialog disconnected.
    EventQueueScope scope;
    setBooleanAttribute(openAttr, true);

    setIsModal(true);

    auto containingBlockBeforeStyleResolution = SingleThreadWeakPtr<RenderBlock> { };
    if (auto* renderer = this->renderer())
        containingBlockBeforeStyleResolution = renderer->containingBlock();

    if (!isInTopLayer())
        addToTopLayer();

    RenderElement::markRendererDirtyAfterTopLayerChange(this->checkedRenderer().get(), containingBlockBeforeStyleResolution.get());

    m_previouslyFocusedElement = document().focusedElement();

    auto hideUntil = topmostPopoverAncestor(TopLayerElementType::Other);

    document().hideAllPopoversUntil(hideUntil, FocusPreviousElement::No, FireEvents::No);

    runFocusingSteps();

    return { };
}

void HTMLDialogElement::close(const String& result)
{
    if (!isOpen())
        return;

    setBooleanAttribute(openAttr, false);

    if (isModal())
        removeFromTopLayer();

    setIsModal(false);

    if (!result.isNull())
        m_returnValue = result;

    if (RefPtr element = std::exchange(m_previouslyFocusedElement, nullptr).get()) {
        FocusOptions options;
        options.preventScroll = true;
        element->focus(options);
    }

    queueTaskToDispatchEvent(TaskSource::UserInteraction, Event::create(eventNames().closeEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

void HTMLDialogElement::requestClose(const String& returnValue)
{
    if (!isOpen())
        return;

    auto cancelEvent = Event::create(eventNames().cancelEvent, Event::CanBubble::No, Event::IsCancelable::Yes);
    dispatchEvent(cancelEvent);
    if (!cancelEvent->defaultPrevented())
        close(returnValue);
}

bool HTMLDialogElement::isValidCommandType(const CommandType command)
{
    return HTMLElement::isValidCommandType(command) || command == CommandType::ShowModal || command == CommandType::Close;
}

bool HTMLDialogElement::handleCommandInternal(const HTMLFormControlElement& invoker, const CommandType& command)
{
    if (HTMLElement::handleCommandInternal(invoker, command))
        return true;

    if (isPopoverShowing())
        return false;

    if (isOpen()) {
        if (command == CommandType::Close) {
            close(nullString());
            return true;
        }
    } else {
        if (command == CommandType::ShowModal) {
            showModal();
            return true;
        }
    }

    return false;
}

void HTMLDialogElement::queueCancelTask()
{
    queueTaskKeepingThisNodeAlive(TaskSource::UserInteraction, [this] {
        auto cancelEvent = Event::create(eventNames().cancelEvent, Event::CanBubble::No, Event::IsCancelable::Yes);
        dispatchEvent(cancelEvent);
        if (!cancelEvent->defaultPrevented())
            close(nullString());
    });
}

// https://html.spec.whatwg.org/multipage/interactive-elements.html#dialog-focusing-steps
void HTMLDialogElement::runFocusingSteps()
{
    RefPtr<Element> control;
    if (hasAttributeWithoutSynchronization(HTMLNames::autofocusAttr))
        control = this;
    if (!control)
        control = findFocusDelegate();

    if (!control)
        control = this;

    RefPtr page = control->document().protectedPage();
    if (!page)
        return;

    if (control->isFocusable())
        control->runFocusingStepsForAutofocus();
    else if (m_isModal)
        document().setFocusedElement(nullptr); // Focus fixup rule

    if (!control->document().isSameOriginAsTopDocument())
        return;

    if (RefPtr mainFrameDocument = control->document().mainFrameDocument())
        mainFrameDocument->clearAutofocusCandidates();
    else
        LOG_ONCE(SiteIsolation, "Unable to fully perform HTMLDialogElement::runFocusingSteps() without access to the main frame document ");
    page->setAutofocusProcessed();
}

bool HTMLDialogElement::supportsFocus() const
{
    return true;
}

void HTMLDialogElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    HTMLElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
    setIsModal(false);
}

void HTMLDialogElement::setIsModal(bool newValue)
{
    if (m_isModal == newValue)
        return;
    Style::PseudoClassChangeInvalidation styleInvalidation(*this, CSSSelector::PseudoClass::Modal, newValue);
    m_isModal = newValue;
}

}
