/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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
#pragma once

#include "HTMLElement.h"

namespace WebCore {

class FormAssociatedCustomElement;

class HTMLMaybeFormAssociatedCustomElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLMaybeFormAssociatedCustomElement);
public:
    static Ref<HTMLMaybeFormAssociatedCustomElement> create(const QualifiedName& tagName, Document&);

    using Node::ref;
    using Node::deref;

    bool isMaybeFormAssociatedCustomElement() const final { return true; }
    bool isFormListedElement() const final;
    bool isValidatedFormListedElement() const final;
    bool isFormAssociatedCustomElement() const;

    FormAssociatedElement* asFormAssociatedElement() final;
    FormListedElement* asFormListedElement() final;
    ValidatedFormListedElement* asValidatedFormListedElement() final;
    FormAssociatedCustomElement* formAssociatedCustomElementForElementInternals() const;

    bool matchesValidPseudoClass() const final;
    bool matchesInvalidPseudoClass() const final;
    bool matchesUserValidPseudoClass() const final;
    bool matchesUserInvalidPseudoClass() const final;

    bool supportsFocus() const final;
    bool isLabelable() const final;
    bool isDisabledFormControl() const final;

    void setInterfaceIsFormAssociated();
    bool hasFormAssociatedInterface() const { return hasEventTargetFlag(EventTargetFlag::HasFormAssociatedCustomElementInterface); }

    void willUpgradeFormAssociated();
    void didUpgradeFormAssociated();

private:
    HTMLMaybeFormAssociatedCustomElement(const QualifiedName& tagName, Document&);
    virtual ~HTMLMaybeFormAssociatedCustomElement();

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void didFinishInsertingNode() final;
    void didMoveToNewDocument(Document&, Document&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void finishParsingChildren() final;
};

static_assert(sizeof(HTMLMaybeFormAssociatedCustomElement) == sizeof(HTMLElement));

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::HTMLMaybeFormAssociatedCustomElement)
    static bool isType(const WebCore::Element& element) { return element.isMaybeFormAssociatedCustomElement(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* element = dynamicDowncast<WebCore::Element>(node);
        return element && isType(*element);
    }
SPECIALIZE_TYPE_TRAITS_END()
