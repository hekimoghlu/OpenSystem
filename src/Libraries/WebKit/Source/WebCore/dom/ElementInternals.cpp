/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#include "ElementInternals.h"
#include "CustomStateSet.h"

#include "AXObjectCache.h"
#include "DocumentInlines.h"
#include "ElementInlines.h"
#include "ElementRareData.h"
#include "HTMLFormElement.h"
#include "HTMLMaybeFormAssociatedCustomElement.h"
#include "ShadowRoot.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace HTMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ElementInternals);

RefPtr<ShadowRoot> ElementInternals::shadowRoot() const
{
    RefPtr element = m_element.get();
    if (!element)
        return nullptr;
    RefPtr shadowRoot = element->shadowRoot();
    if (!shadowRoot)
        return nullptr;
    if (!shadowRoot->isAvailableToElementInternals())
        return nullptr;
    return shadowRoot;
}

ExceptionOr<RefPtr<HTMLFormElement>> ElementInternals::form() const
{
    if (RefPtr element = elementAsFormAssociatedCustom())
        return element->form();
    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<void> ElementInternals::setFormValue(CustomElementFormValue&& value, std::optional<CustomElementFormValue>&& state)
{
    if (RefPtr element = elementAsFormAssociatedCustom()) {
        element->setFormValue(WTFMove(value), WTFMove(state));
        return { };
    }

    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<void> ElementInternals::setValidity(ValidityStateFlags validityStateFlags, String&& message, HTMLElement* validationAnchor)
{
    if (RefPtr element = elementAsFormAssociatedCustom())
        return element->setValidity(validityStateFlags, WTFMove(message), validationAnchor);
    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<bool> ElementInternals::willValidate() const
{
    if (RefPtr element = elementAsFormAssociatedCustom())
        return element->willValidate();
    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<RefPtr<ValidityState>> ElementInternals::validity()
{
    if (RefPtr element = elementAsFormAssociatedCustom())
        return element->validity();
    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<String> ElementInternals::validationMessage() const
{
    if (RefPtr element = elementAsFormAssociatedCustom())
        return element->validationMessage();
    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<bool> ElementInternals::reportValidity()
{
    if (RefPtr element = elementAsFormAssociatedCustom())
        return element->reportValidity();
    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<bool> ElementInternals::checkValidity()
{
    if (RefPtr element = elementAsFormAssociatedCustom())
        return element->checkValidity();
    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<RefPtr<NodeList>> ElementInternals::labels()
{
    if (RefPtr element = elementAsFormAssociatedCustom())
        return element->asHTMLElement().labels();
    return Exception { ExceptionCode::NotSupportedError };
}

FormAssociatedCustomElement* ElementInternals::elementAsFormAssociatedCustom() const
{
    if (RefPtr element = dynamicDowncast<HTMLMaybeFormAssociatedCustomElement>(m_element.get()))
        return element->formAssociatedCustomElementForElementInternals();
    return nullptr;
}

static const AtomString& computeValueForAttribute(Element& element, const QualifiedName& name)
{
    auto& value = element.attributeWithoutSynchronization(name);
    if (CheckedPtr defaultARIA = element.customElementDefaultARIAIfExists(); value.isNull() && defaultARIA)
        return defaultARIA->valueForAttribute(element, name);
    return value;
}

void ElementInternals::setAttributeWithoutSynchronization(const QualifiedName& name, const AtomString& value)
{
    RefPtr element = m_element.get();
    auto oldValue = computeValueForAttribute(*element, name);

    element->checkedCustomElementDefaultARIA()->setValueForAttribute(name, value);

    if (CheckedPtr cache = element->document().existingAXObjectCache())
        cache->deferAttributeChangeIfNeeded(*element, name, oldValue, computeValueForAttribute(*element, name));
}

const AtomString& ElementInternals::attributeWithoutSynchronization(const QualifiedName& name) const
{
    RefPtr element = m_element.get();
    CheckedPtr defaultARIA = element->customElementDefaultARIAIfExists();
    return defaultARIA ? defaultARIA->valueForAttribute(*element, name) : nullAtom();
}

RefPtr<Element> ElementInternals::getElementAttribute(const QualifiedName& name) const
{
    RefPtr element = m_element.get();
    CheckedPtr defaultARIA = m_element->customElementDefaultARIAIfExists();
    return defaultARIA ? defaultARIA->elementForAttribute(*element, name) : nullptr;
}

void ElementInternals::setElementAttribute(const QualifiedName& name, Element* value)
{
    RefPtr element = m_element.get();
    auto oldValue = computeValueForAttribute(*element, name);

    element->checkedCustomElementDefaultARIA()->setElementForAttribute(name, value);

    if (CheckedPtr cache = element->document().existingAXObjectCache())
        cache->deferAttributeChangeIfNeeded(*element, name, oldValue, computeValueForAttribute(*element, name));
}

std::optional<Vector<Ref<Element>>> ElementInternals::getElementsArrayAttribute(const QualifiedName& name) const
{
    RefPtr element = m_element.get();
    CheckedPtr defaultARIA = m_element->customElementDefaultARIAIfExists();
    if (!defaultARIA)
        return std::nullopt;
    return defaultARIA->elementsForAttribute(*element, name);
}

void ElementInternals::setElementsArrayAttribute(const QualifiedName& name, std::optional<Vector<Ref<Element>>>&& value)
{
    RefPtr element = m_element.get();
    auto oldValue = computeValueForAttribute(*element, name);

    element->checkedCustomElementDefaultARIA()->setElementsForAttribute(name, WTFMove(value));

    if (CheckedPtr cache = element->document().existingAXObjectCache())
        cache->deferAttributeChangeIfNeeded(*element, name, oldValue, computeValueForAttribute(*element, name));
}

CustomStateSet& ElementInternals::states()
{
    return m_element->ensureCustomStateSet();
}

} // namespace WebCore
