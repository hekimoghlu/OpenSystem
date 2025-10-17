/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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

#include "ActiveDOMObject.h"
#include "AttachmentAssociatedElement.h"
#include "HTMLElement.h"
#include "MediaQuery.h"
#include "Timer.h"

namespace WebCore {

class HTMLSourceElement final
    : public HTMLElement
#if ENABLE(ATTACHMENT_ELEMENT)
    , public AttachmentAssociatedElement
#endif
    , public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLSourceElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLSourceElement);
public:
    static Ref<HTMLSourceElement> create(Document&);
    static Ref<HTMLSourceElement> create(const QualifiedName&, Document&);

    // ActiveDOMObject.
    void ref() const final { HTMLElement::ref(); }
    void deref() const final { HTMLElement::deref(); }

    void scheduleErrorEvent();
    void cancelPendingErrorEvent();

    const MQ::MediaQueryList& parsedMediaAttribute(Document&) const;

private:
    HTMLSourceElement(const QualifiedName&, Document&);
    
    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;
    void didMoveToNewDocument(Document& oldDocument, Document& newDocument) final;

    bool isURLAttribute(const Attribute&) const final;
    bool attributeContainsURL(const Attribute&) const final;
    Attribute replaceURLsInAttributeValue(const Attribute&, const UncheckedKeyHashMap<String, String>&) const override;
    void addCandidateSubresourceURLs(ListHashSet<URL>&) const override;

    // ActiveDOMObject.
    void stop() final;

#if ENABLE(ATTACHMENT_ELEMENT)
    HTMLSourceElement& asHTMLElement() final { return *this; }
    const HTMLSourceElement& asHTMLElement() const final { return *this; }

    void refAttachmentAssociatedElement() const final { HTMLElement::ref(); }
    void derefAttachmentAssociatedElement() const final { HTMLElement::deref(); }

    AttachmentAssociatedElement* asAttachmentAssociatedElement() final { return this; }

    AttachmentAssociatedElementType attachmentAssociatedElementType() const final { return AttachmentAssociatedElementType::Source; }
#endif

    Ref<Element> cloneElementWithoutAttributesAndChildren(TreeScope&) final;
    void copyNonAttributePropertiesFromElement(const Element&) final;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;

    TaskCancellationGroup m_errorEventCancellationGroup;
    bool m_shouldCallSourcesChanged { false };
    mutable std::optional<MQ::MediaQueryList> m_cachedParsedMediaAttribute;
};

} // namespace WebCore
