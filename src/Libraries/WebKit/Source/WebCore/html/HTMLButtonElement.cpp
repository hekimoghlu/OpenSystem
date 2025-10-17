/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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
#include "HTMLButtonElement.h"

#include "CommonAtomStrings.h"
#include "DOMFormData.h"
#include "ElementInlines.h"
#include "EventNames.h"
#include "HTMLFormElement.h"
#include "HTMLNames.h"
#include "KeyboardEvent.h"
#include "RenderButton.h"
#include <wtf/SetForScope.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(SERVICE_CONTROLS)
#include "ImageControlsMac.h"
#endif

#if ENABLE(SPATIAL_IMAGE_CONTROLS)
#include "SpatialImageControls.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLButtonElement);

using namespace HTMLNames;

inline HTMLButtonElement::HTMLButtonElement(const QualifiedName& tagName, Document& document, HTMLFormElement* form)
    : HTMLFormControlElement(tagName, document, form)
    , m_type(SUBMIT)
    , m_isActivatedSubmit(false)
{
    ASSERT(hasTagName(buttonTag));
}

Ref<HTMLButtonElement> HTMLButtonElement::create(const QualifiedName& tagName, Document& document, HTMLFormElement* form)
{
    return adoptRef(*new HTMLButtonElement(tagName, document, form));
}

Ref<HTMLButtonElement> HTMLButtonElement::create(Document& document)
{
    return adoptRef(*new HTMLButtonElement(buttonTag, document, nullptr));
}

void HTMLButtonElement::setType(const AtomString& type)
{
    setAttributeWithoutSynchronization(typeAttr, type);
}

RenderPtr<RenderElement> HTMLButtonElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition& position)
{
    // https://html.spec.whatwg.org/multipage/rendering.html#button-layout
    DisplayType display = style.display();
    if (display == DisplayType::InlineGrid || display == DisplayType::Grid || display == DisplayType::InlineFlex || display == DisplayType::Flex)
        return HTMLFormControlElement::createElementRenderer(WTFMove(style), position);
    return createRenderer<RenderButton>(*this, WTFMove(style));
}

int HTMLButtonElement::defaultTabIndex() const
{
    return 0;
}

const AtomString& HTMLButtonElement::formControlType() const
{
    switch (m_type) {
    case SUBMIT:
        return submitAtom();
    case BUTTON:
        return HTMLNames::buttonTag->localName();
    case RESET:
        return resetAtom();
    }

    ASSERT_NOT_REACHED();
    return emptyAtom();
}

bool HTMLButtonElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    if (name == alignAttr) {
        // Don't map 'align' attribute.  This matches what Firefox and IE do, but not Opera.
        // See http://bugs.webkit.org/show_bug.cgi?id=12071
        return false;
    }

    return HTMLFormControlElement::hasPresentationalHintsForAttribute(name);
}

void HTMLButtonElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == typeAttr) {
        Type oldType = m_type;
        if (equalLettersIgnoringASCIICase(newValue, "reset"_s))
            m_type = RESET;
        else if (equalLettersIgnoringASCIICase(newValue, "button"_s))
            m_type = BUTTON;
        else
            m_type = SUBMIT;
        if (oldType != m_type) {
            updateWillValidateAndValidity();
            if (form() && (oldType == SUBMIT || m_type == SUBMIT))
                form()->resetDefaultButton();
        }
    } else
        HTMLFormControlElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void HTMLButtonElement::defaultEventHandler(Event& event)
{
#if ENABLE(SERVICE_CONTROLS)
    if (ImageControlsMac::handleEvent(*this, event))
        return;
#endif
#if ENABLE(SPATIAL_IMAGE_CONTROLS)
    if (SpatialImageControls::handleEvent(*this, event))
        return;
#endif
    auto& eventNames = WebCore::eventNames();
    if (event.type() == eventNames.DOMActivateEvent && !isDisabledFormControl()) {
        RefPtr<HTMLFormElement> protectedForm(form());

        if (commandForElement()) {
            if (m_type != BUTTON && form())
                return;

            handleCommand();

        } else if (protectedForm) {
            // Update layout before processing form actions in case the style changes
            // the Form or button relationships.
            protectedDocument()->updateLayoutIgnorePendingStylesheets();

            if (RefPtr currentForm = form()) {
                if (m_type == SUBMIT)
                    currentForm->submitIfPossible(&event, this);

                if (m_type == RESET)
                    currentForm->reset();
            }

            if (m_type == SUBMIT || m_type == RESET)
                event.setDefaultHandled();
        }

        if (!(protectedForm && m_type == SUBMIT))
            handlePopoverTargetAction(event.target());

    }

    if (RefPtr keyboardEvent = dynamicDowncast<KeyboardEvent>(event)) {
        if (keyboardEvent->type() == eventNames.keydownEvent && keyboardEvent->keyIdentifier() == "U+0020"_s) {
            setActive(true);
            // No setDefaultHandled() - IE dispatches a keypress in this case.
            return;
        }
        if (keyboardEvent->type() == eventNames.keypressEvent) {
            switch (keyboardEvent->charCode()) {
                case '\r':
                    dispatchSimulatedClick(keyboardEvent.get());
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
                dispatchSimulatedClick(keyboardEvent.get());
            keyboardEvent->setDefaultHandled();
            return;
        }
    }

    HTMLFormControlElement::defaultEventHandler(event);
}

bool HTMLButtonElement::willRespondToMouseClickEventsWithEditability(Editability) const
{
    return !isDisabledFormControl();
}

bool HTMLButtonElement::isSuccessfulSubmitButton() const
{
    // HTML spec says that buttons must have names to be considered successful.
    // However, other browsers do not impose this constraint.
    return m_type == SUBMIT;
}

bool HTMLButtonElement::matchesDefaultPseudoClass() const
{
    return isSuccessfulSubmitButton() && form() && form()->defaultButton() == this;
}

bool HTMLButtonElement::isActivatedSubmit() const
{
    return m_isActivatedSubmit;
}

void HTMLButtonElement::setActivatedSubmit(bool flag)
{
    m_isActivatedSubmit = flag;
}

bool HTMLButtonElement::appendFormData(DOMFormData& formData)
{
    if (m_type != SUBMIT || name().isEmpty() || !m_isActivatedSubmit)
        return false;
    formData.append(name(), value());
    return true;
}

bool HTMLButtonElement::isURLAttribute(const Attribute& attribute) const
{
    return attribute.name() == formactionAttr || HTMLFormControlElement::isURLAttribute(attribute);
}

const AtomString& HTMLButtonElement::value() const
{
    return attributeWithoutSynchronization(valueAttr);
}

bool HTMLButtonElement::computeWillValidate() const
{
    return m_type == SUBMIT && HTMLFormControlElement::computeWillValidate();
}

bool HTMLButtonElement::isSubmitButton() const
{
    return m_type == SUBMIT;
}

bool HTMLButtonElement::isExplicitlySetSubmitButton() const
{
    return isSubmitButton() && hasAttributeWithoutSynchronization(HTMLNames::typeAttr);
}

} // namespace
