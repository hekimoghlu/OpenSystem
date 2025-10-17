/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
#include "HTMLMaybeFormAssociatedCustomElement.h"

#include "Document.h"
#include "ElementRareData.h"
#include "FormAssociatedCustomElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLMaybeFormAssociatedCustomElement);

using namespace HTMLNames;

HTMLMaybeFormAssociatedCustomElement::HTMLMaybeFormAssociatedCustomElement(const QualifiedName& tagName, Document& document)
    : HTMLElement { tagName, document, TypeFlag::HasDidMoveToNewDocument }
{
    ASSERT(Document::validateCustomElementName(tagName.localName()) == CustomElementNameValidationStatus::Valid);
}

HTMLMaybeFormAssociatedCustomElement::~HTMLMaybeFormAssociatedCustomElement() = default;

Ref<HTMLMaybeFormAssociatedCustomElement> HTMLMaybeFormAssociatedCustomElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLMaybeFormAssociatedCustomElement(tagName, document));
}

bool HTMLMaybeFormAssociatedCustomElement::isFormListedElement() const
{
    return isFormAssociatedCustomElement();
}

bool HTMLMaybeFormAssociatedCustomElement::isValidatedFormListedElement() const
{
    return isFormAssociatedCustomElement();
}

bool HTMLMaybeFormAssociatedCustomElement::isFormAssociatedCustomElement() const
{
    return hasFormAssociatedInterface() && isDefinedCustomElement();
}

FormAssociatedElement* HTMLMaybeFormAssociatedCustomElement::asFormAssociatedElement()
{
    return isFormAssociatedCustomElement() ? &formAssociatedCustomElementUnsafe() : nullptr;
}

FormListedElement* HTMLMaybeFormAssociatedCustomElement::asFormListedElement()
{
    return isFormAssociatedCustomElement() ? &formAssociatedCustomElementUnsafe() : nullptr;
}

ValidatedFormListedElement* HTMLMaybeFormAssociatedCustomElement::asValidatedFormListedElement()
{
    return isFormAssociatedCustomElement() ? &formAssociatedCustomElementUnsafe() : nullptr;
}

FormAssociatedCustomElement* HTMLMaybeFormAssociatedCustomElement::formAssociatedCustomElementForElementInternals() const
{
    return hasFormAssociatedInterface() && isPrecustomizedOrDefinedCustomElement() ? &formAssociatedCustomElementUnsafe() : nullptr;
}

bool HTMLMaybeFormAssociatedCustomElement::matchesValidPseudoClass() const
{
    return isFormAssociatedCustomElement() && formAssociatedCustomElementUnsafe().matchesValidPseudoClass();
}

bool HTMLMaybeFormAssociatedCustomElement::matchesInvalidPseudoClass() const
{
    return isFormAssociatedCustomElement() && formAssociatedCustomElementUnsafe().matchesInvalidPseudoClass();
}

bool HTMLMaybeFormAssociatedCustomElement::matchesUserValidPseudoClass() const
{
    return isFormAssociatedCustomElement() && formAssociatedCustomElementUnsafe().matchesUserValidPseudoClass();
}

bool HTMLMaybeFormAssociatedCustomElement::matchesUserInvalidPseudoClass() const
{
    return isFormAssociatedCustomElement() && formAssociatedCustomElementUnsafe().matchesUserInvalidPseudoClass();
}

bool HTMLMaybeFormAssociatedCustomElement::supportsFocus() const
{
    return isFormAssociatedCustomElement() ? (shadowRoot() && shadowRoot()->delegatesFocus()) || (HTMLElement::supportsFocus() && !formAssociatedCustomElementUnsafe().isDisabled()) : HTMLElement::supportsFocus();
}

bool HTMLMaybeFormAssociatedCustomElement::isLabelable() const
{
    return isFormAssociatedCustomElement();
}

bool HTMLMaybeFormAssociatedCustomElement::isDisabledFormControl() const
{
    return isFormAssociatedCustomElement() && formAssociatedCustomElementUnsafe().isDisabled();
}

Node::InsertedIntoAncestorResult HTMLMaybeFormAssociatedCustomElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    HTMLElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (isFormAssociatedCustomElement())
        formAssociatedCustomElementUnsafe().insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (!insertionType.connectedToDocument)
        return InsertedIntoAncestorResult::Done;
    return InsertedIntoAncestorResult::NeedsPostInsertionCallback;
}

void HTMLMaybeFormAssociatedCustomElement::didFinishInsertingNode()
{
    HTMLElement::didFinishInsertingNode();
    if (isFormAssociatedCustomElement())
        formAssociatedCustomElementUnsafe().didFinishInsertingNode();
}

void HTMLMaybeFormAssociatedCustomElement::didMoveToNewDocument(Document& oldDocument, Document& newDocument)
{
    HTMLElement::didMoveToNewDocument(oldDocument, newDocument);
    if (isFormAssociatedCustomElement())
        formAssociatedCustomElementUnsafe().didMoveToNewDocument();
}

void HTMLMaybeFormAssociatedCustomElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    HTMLElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
    if (isFormAssociatedCustomElement())
        formAssociatedCustomElementUnsafe().removedFromAncestor(removalType, oldParentOfRemovedTree);
}

void HTMLMaybeFormAssociatedCustomElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    HTMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
    if (isFormAssociatedCustomElement())
        formAssociatedCustomElementUnsafe().parseAttribute(name, newValue);
}

void HTMLMaybeFormAssociatedCustomElement::finishParsingChildren()
{
    HTMLElement::finishParsingChildren();
    if (isFormAssociatedCustomElement())
        formAssociatedCustomElementUnsafe().finishParsingChildren();
}

void HTMLMaybeFormAssociatedCustomElement::setInterfaceIsFormAssociated()
{
    setEventTargetFlag(EventTargetFlag::HasFormAssociatedCustomElementInterface, true);
    ensureFormAssociatedCustomElement();
}

void HTMLMaybeFormAssociatedCustomElement::willUpgradeFormAssociated()
{
    ASSERT(isPrecustomizedCustomElement());
    setInterfaceIsFormAssociated();
    formAssociatedCustomElementUnsafe().willUpgrade();
}

void HTMLMaybeFormAssociatedCustomElement::didUpgradeFormAssociated()
{
    ASSERT(isFormAssociatedCustomElement());
    formAssociatedCustomElementUnsafe().didUpgrade();
}

} // namespace WebCore
