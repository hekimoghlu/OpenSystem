/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
#include "HTMLSummaryElement.h"

#include "ElementInlines.h"
#include "EventNames.h"
#include "HTMLDetailsElement.h"
#include "HTMLFormControlElement.h"
#include "HTMLSlotElement.h"
#include "KeyboardEvent.h"
#include "MouseEvent.h"
#include "PlatformMouseEvent.h"
#include "RenderBlockFlow.h"
#include "SVGAElement.h"
#include "SVGElementTypeHelpers.h"
#include "ShadowRoot.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLSummaryElement);

using namespace HTMLNames;

Ref<HTMLSummaryElement> HTMLSummaryElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLSummaryElement(tagName, document));
}

HTMLSummaryElement::HTMLSummaryElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(summaryTag));
}

RefPtr<HTMLDetailsElement> HTMLSummaryElement::detailsElement() const
{
    if (auto* parent = dynamicDowncast<HTMLDetailsElement>(parentElement()))
        return parent;
    // Fallback summary element is in the shadow tree.
    if (auto* details = dynamicDowncast<HTMLDetailsElement>(shadowHost()))
        return details;
    return nullptr;
}

bool HTMLSummaryElement::isActiveSummary() const
{
    RefPtr<HTMLDetailsElement> details = detailsElement();
    if (!details)
        return false;
    return details->isActiveSummary(*this);
}

static bool isInSummaryInteractiveContent(EventTarget* target)
{
    for (RefPtr element = dynamicDowncast<Element>(target); element && !is<HTMLSummaryElement>(element); element = element->parentOrShadowHostElement()) {
        auto* htmlElement = dynamicDowncast<HTMLElement>(*element);
        if ((htmlElement && htmlElement->isInteractiveContent()) || is<SVGAElement>(element))
            return true;
    }
    return false;
}

int HTMLSummaryElement::defaultTabIndex() const
{
    return isActiveSummary() ? 0 : -1;
}

bool HTMLSummaryElement::supportsFocus() const
{
    return isActiveSummary() || HTMLElement::supportsFocus();
}

void HTMLSummaryElement::defaultEventHandler(Event& event)
{
    if (isActiveSummary()) {
        auto& eventNames = WebCore::eventNames();
        if (event.type() == eventNames.DOMActivateEvent && !isInSummaryInteractiveContent(event.target())) {
            if (RefPtr<HTMLDetailsElement> details = detailsElement())
                details->toggleOpen();
            event.setDefaultHandled();
            return;
        }

        if (auto* keyboardEvent = dynamicDowncast<KeyboardEvent>(event)) {
            if (keyboardEvent->type() == eventNames.keydownEvent && keyboardEvent->keyIdentifier() == "U+0020"_s) {
                setActive(true);
                // No setDefaultHandled() - IE dispatches a keypress in this case.
                return;
            }
            if (keyboardEvent->type() == eventNames.keypressEvent) {
                switch (keyboardEvent->charCode()) {
                case '\r':
                    dispatchSimulatedClick(&event);
                    keyboardEvent->setDefaultHandled();
                    return;
                case ' ':
                    // Prevent scrolling down the page.
                    keyboardEvent->setDefaultHandled();
                    return;
                }
            }
            if (keyboardEvent->type() == eventNames.keyupEvent && keyboardEvent->keyIdentifier() == "U+0020"_s) {
                if (active())
                    dispatchSimulatedClick(&event);
                keyboardEvent->setDefaultHandled();
                return;
            }
        }
    }

    HTMLElement::defaultEventHandler(event);
}

bool HTMLSummaryElement::willRespondToMouseClickEventsWithEditability(Editability editability) const
{
    return isActiveSummary() || HTMLElement::willRespondToMouseClickEventsWithEditability(editability);
}

}
