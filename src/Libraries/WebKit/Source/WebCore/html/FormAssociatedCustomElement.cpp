/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
#include "FormAssociatedCustomElement.h"

#include "CustomElementReactionQueue.h"
#include "ElementAncestorIteratorInlines.h"
#include "HTMLFieldSetElement.h"
#include "HTMLFormElement.h"
#include "NodeRareData.h"
#include "ValidationMessage.h"
#include <wtf/Ref.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FormAssociatedCustomElement);

using namespace HTMLNames;

FormAssociatedCustomElement::FormAssociatedCustomElement(HTMLMaybeFormAssociatedCustomElement& element)
    : ValidatedFormListedElement { nullptr }
    , m_element { element }
{
}

FormAssociatedCustomElement::~FormAssociatedCustomElement()
{
    clearForm();
}

Ref<FormAssociatedCustomElement> FormAssociatedCustomElement::create(HTMLMaybeFormAssociatedCustomElement& element)
{
    return adoptRef(*new FormAssociatedCustomElement(element));
}

ExceptionOr<void> FormAssociatedCustomElement::setValidity(ValidityStateFlags validityStateFlags, String&& message, HTMLElement* validationAnchor)
{
    ASSERT(m_element->isPrecustomizedOrDefinedCustomElement());

    if (!validityStateFlags.isValid() && message.isEmpty())
        return Exception { ExceptionCode::TypeError };

    m_validityStateFlags = validityStateFlags;
    setCustomValidity(validityStateFlags.isValid() ? emptyString() : WTFMove(message));

    if (validationAnchor && !validationAnchor->isDescendantOrShadowDescendantOf(*m_element))
        return Exception { ExceptionCode::NotFoundError };

    m_validationAnchor = validationAnchor;

    return { };
}

String FormAssociatedCustomElement::validationMessage() const
{
    ASSERT(m_element->isPrecustomizedOrDefinedCustomElement());
    return customValidationMessage();
}

ALWAYS_INLINE static CustomElementFormValue cloneIfIsFormData(CustomElementFormValue&& value)
{
    return WTF::switchOn(WTFMove(value), [](RefPtr<DOMFormData> value) -> CustomElementFormValue {
        return value->clone().ptr();
    }, [](const auto& value) -> CustomElementFormValue {
        return value;
    });
}

void FormAssociatedCustomElement::setFormValue(CustomElementFormValue&& submissionValue, std::optional<CustomElementFormValue>&& state)
{
    ASSERT(m_element->isPrecustomizedOrDefinedCustomElement());

    m_submissionValue = cloneIfIsFormData(WTFMove(submissionValue));
    m_state = state.has_value() ? cloneIfIsFormData(WTFMove(state.value())) : m_submissionValue;
}

HTMLElement* FormAssociatedCustomElement::validationAnchorElement()
{
    ASSERT(m_element->isDefinedCustomElement());
    return m_validationAnchor.get();
}

bool FormAssociatedCustomElement::computeValidity() const
{
    ASSERT(m_element->isPrecustomizedOrDefinedCustomElement());
    return m_validityStateFlags.isValid();
}

bool FormAssociatedCustomElement::appendFormData(DOMFormData& formData)
{
    ASSERT(m_element->isDefinedCustomElement());

    WTF::switchOn(m_submissionValue, [&](RefPtr<DOMFormData> value) {
        for (const auto& item : value->items()) {
            WTF::switchOn(item.data, [&](const String& value) {
                formData.append(item.name, value);
            }, [&](RefPtr<File> value) {
                formData.append(item.name, *value);
            });
        }
    }, [&](const String& value) {
        if (!name().isEmpty())
            formData.append(name(), value);
    }, [&](RefPtr<File> value) {
        if (!name().isEmpty())
            formData.append(name(), *value);
    }, [](std::nullptr_t) {
        // do nothing
    });

    return true;
}

bool FormAssociatedCustomElement::isEnumeratable() const
{
    ASSERT(m_element->isDefinedCustomElement());
    return true;
}

void FormAssociatedCustomElement::reset()
{
    ASSERT(m_element->isDefinedCustomElement());
    setInteractedWithSinceLastFormSubmitEvent(false);
    CustomElementReactionQueue::enqueueFormResetCallbackIfNeeded(*m_element);
}

void FormAssociatedCustomElement::disabledStateChanged()
{
    ASSERT(m_element->isDefinedCustomElement());
    ValidatedFormListedElement::disabledStateChanged();
    CustomElementReactionQueue::enqueueFormDisabledCallbackIfNeeded(*m_element, isDisabled());
}

void FormAssociatedCustomElement::didChangeForm()
{
    ASSERT(m_element->isDefinedCustomElement());
    ValidatedFormListedElement::didChangeForm();
    if (!belongsToFormThatIsBeingDestroyed())
        CustomElementReactionQueue::enqueueFormAssociatedCallbackIfNeeded(*m_element, form());
}

void FormAssociatedCustomElement::willUpgrade()
{
    setDataListAncestorState(TriState::False);
}

void FormAssociatedCustomElement::didUpgrade()
{
    ASSERT(!form());

    HTMLElement& element = asHTMLElement();

    parseFormAttribute(element.attributeWithoutSynchronization(formAttr));
    parseDisabledAttribute(element.attributeWithoutSynchronization(disabledAttr));
    parseReadOnlyAttribute(element.attributeWithoutSynchronization(readonlyAttr));

    setDataListAncestorState(TriState::Indeterminate);
    updateWillValidateAndValidity();
    syncWithFieldsetAncestors(element.parentNode());
    invalidateElementsCollectionCachesInAncestors();
    restoreFormControlStateIfNecessary();
}

void FormAssociatedCustomElement::finishParsingChildren()
{
    if (!FormController::ownerForm(*this))
        restoreFormControlStateIfNecessary();
}

void FormAssociatedCustomElement::invalidateElementsCollectionCachesInAncestors()
{
    auto invalidateElementsCache = [](HTMLElement& element) {
        if (auto* nodeLists = element.nodeLists()) {
            // We kick the "form" attribute to invalidate only the FormControls, FieldSetElements,
            // and RadioNodeList collections, and do so without duplicating invalidation logic of Node.cpp.
            nodeLists->invalidateCachesForAttribute(HTMLNames::formAttr);
        }
    };

    if (RefPtr form = this->form())
        invalidateElementsCache(*form);

    for (auto& ancestor : lineageOfType<HTMLFieldSetElement>(*m_element))
        invalidateElementsCache(ancestor);
}

const AtomString& FormAssociatedCustomElement::formControlType() const
{
    return asHTMLElement().localName();
}

bool FormAssociatedCustomElement::shouldSaveAndRestoreFormControlState() const
{
    const auto& element = asHTMLElement();
    ASSERT(element.reactionQueue());
    return element.isDefinedCustomElement() && element.reactionQueue()->hasFormStateRestoreCallback();
}

FormControlState FormAssociatedCustomElement::saveFormControlState() const
{
    ASSERT(m_element->isDefinedCustomElement());

    FormControlState savedState;

    // FIXME: Support File when saving / restoring state.
    // https://bugs.webkit.org/show_bug.cgi?id=249895
    bool didLogMessage = false;
    auto logUnsupportedFileWarning = [&](RefPtr<File>) {
        Ref document = asHTMLElement().document();
        if (document->frame() && !didLogMessage) {
            document->addConsoleMessage(MessageSource::JS, MessageLevel::Warning, "File isn't currently supported when saving / restoring state."_s);
            didLogMessage = true;
        }
    };

    WTF::switchOn(m_state, [&](RefPtr<DOMFormData> state) {
        savedState.reserveInitialCapacity(state->items().size() * 2);

        for (const auto& item : state->items()) {
            WTF::switchOn(item.data, [&](const String& value) {
                savedState.append(item.name);
                savedState.append(value);
            }, logUnsupportedFileWarning);
        }

        savedState.shrinkToFit();
    }, [&](const String& state) {
        savedState.append(state);
    }, [](std::nullptr_t) {
        // do nothing
    }, logUnsupportedFileWarning);

    return savedState;
}

void FormAssociatedCustomElement::restoreFormControlState(const FormControlState& savedState)
{
    ASSERT(m_element->isDefinedCustomElement());

    CustomElementFormValue restoredState;

    if (savedState.size() == 1)
        restoredState.emplace<String>(savedState[0]);
    else {
        auto formData = DOMFormData::create(&asHTMLElement().document(), PAL::UTF8Encoding());
        for (size_t i = 0; i < savedState.size(); i += 2)
            formData->append(savedState[i], savedState[i + 1]);
        restoredState.emplace<RefPtr<DOMFormData>>(formData.ptr());
    }

    CustomElementReactionQueue::enqueueFormStateRestoreCallbackIfNeeded(*m_element, WTFMove(restoredState));
}

} // namespace Webcore
