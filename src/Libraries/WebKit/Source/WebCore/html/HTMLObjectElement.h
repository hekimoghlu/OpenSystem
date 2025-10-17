/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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

#include "FormListedElement.h"
#include "HTMLPlugInImageElement.h"

namespace WebCore {

class HTMLFormElement;

class HTMLObjectElement final : public HTMLPlugInImageElement, public FormListedElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLObjectElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLObjectElement);
public:
    static Ref<HTMLObjectElement> create(const QualifiedName&, Document&, HTMLFormElement*);

    bool isExposed() const { return m_isExposed; }

    bool hasFallbackContent() const;
    bool useFallbackContent() const final { return m_useFallbackContent; }
    void renderFallbackContent();

    bool willValidate() const final { return false; }

    // Implementation of constraint validation API.
    // Note that the object elements are always barred from constraint validation.
    static bool checkValidity() { return true; }
    static bool reportValidity() { return true; }

    using HTMLPlugInImageElement::ref;
    using HTMLPlugInImageElement::deref;

private:
    HTMLObjectElement(const QualifiedName&, Document&, HTMLFormElement*);
    ~HTMLObjectElement();

    int defaultTabIndex() const final;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    bool hasPresentationalHintsForAttribute(const QualifiedName&) const final;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) final;

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void didFinishInsertingNode() final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;

    void didMoveToNewDocument(Document& oldDocument, Document& newDocument) final;

    void childrenChanged(const ChildChange&) final;

    bool isURLAttribute(const Attribute&) const final;
    const AtomString& imageSourceURL() const final;

    void addSubresourceAttributeURLs(ListHashSet<URL>&) const final;

    void updateWidget(CreatePlugins) final;
    void updateExposedState();

    // FIXME: Better share code between <object> and <embed>.
    void parametersForPlugin(Vector<AtomString>& paramNames, Vector<AtomString>& paramValues);

    void refFormAssociatedElement() const final { ref(); }
    void derefFormAssociatedElement() const final { deref(); }

    FormAssociatedElement* asFormAssociatedElement() final { return this; }
    FormListedElement* asFormListedElement() final { return this; }
    ValidatedFormListedElement* asValidatedFormListedElement() final { return nullptr; }

    // These functions can be called concurrently for ValidityState.
    HTMLObjectElement& asHTMLElement() final { return *this; }
    const HTMLObjectElement& asHTMLElement() const final { return *this; }

    bool isFormListedElement() const final { return true; }
    bool isValidatedFormListedElement() const final { return false; }

    bool isEnumeratable() const final { return true; }

    bool canContainRangeEndPoint() const final;

    bool m_isExposed { true };
    bool m_useFallbackContent { false };
};

} // namespace WebCore
