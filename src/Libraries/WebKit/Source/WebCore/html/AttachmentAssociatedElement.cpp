/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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
#include "AttachmentAssociatedElement.h"

#if ENABLE(ATTACHMENT_ELEMENT)

#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "ElementChildIteratorInlines.h"
#include "HTMLAttachmentElement.h"
#include "ShadowRoot.h"

namespace WebCore {

void AttachmentAssociatedElement::setAttachmentElement(Ref<HTMLAttachmentElement>&& attachment)
{
    if (auto existingAttachment = attachmentElement())
        existingAttachment->remove();

    attachment->setInlineStyleProperty(CSSPropertyDisplay, CSSValueNone, IsImportant::Yes);
    asHTMLElement().ensureUserAgentShadowRoot().appendChild(WTFMove(attachment));
}

RefPtr<HTMLAttachmentElement> AttachmentAssociatedElement::attachmentElement() const
{
    if (RefPtr shadowRoot = asHTMLElement().userAgentShadowRoot())
        return childrenOfType<HTMLAttachmentElement>(*shadowRoot).first();

    return nullptr;
}

const String& AttachmentAssociatedElement::attachmentIdentifier() const
{
    if (!m_pendingClonedAttachmentID.isEmpty())
        return m_pendingClonedAttachmentID;

    if (auto attachment = attachmentElement())
        return attachment->uniqueIdentifier();

    return nullAtom();
}

void AttachmentAssociatedElement::didUpdateAttachmentIdentifier()
{
    m_pendingClonedAttachmentID = { };
}

void AttachmentAssociatedElement::copyAttachmentAssociatedPropertiesFromElement(const AttachmentAssociatedElement& source)
{
    m_pendingClonedAttachmentID = !source.m_pendingClonedAttachmentID.isEmpty() ? source.m_pendingClonedAttachmentID : source.attachmentIdentifier();
}

void AttachmentAssociatedElement::cloneAttachmentAssociatedElementWithoutAttributesAndChildren(AttachmentAssociatedElement& clone, Document& targetDocument)
{
    if (auto attachment = attachmentElement()) {
        auto attachmentClone = attachment->cloneElementWithoutChildren(targetDocument);
        clone.setAttachmentElement(downcast<HTMLAttachmentElement>(attachmentClone.get()));
    }
}

} // namespace WebCore

#endif // ENABLE(ATTACHMENT_ELEMENT)
