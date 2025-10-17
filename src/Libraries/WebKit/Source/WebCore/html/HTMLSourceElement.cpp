/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
#include "HTMLSourceElement.h"

#include "ElementInlines.h"
#include "Event.h"
#include "EventNames.h"
#include "HTMLImageElement.h"
#include "HTMLNames.h"
#include "HTMLPictureElement.h"
#include "HTMLSrcsetParser.h"
#include "Logging.h"
#include "MediaQueryParser.h"
#include "MediaQueryParserContext.h"
#include "NodeName.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(VIDEO)
#include "HTMLMediaElement.h"
#endif

#if ENABLE(MODEL_ELEMENT)
#include "HTMLModelElement.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLSourceElement);

using namespace HTMLNames;

inline HTMLSourceElement::HTMLSourceElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document, TypeFlag::HasDidMoveToNewDocument)
    , ActiveDOMObject(document)
{
    LOG(Media, "HTMLSourceElement::HTMLSourceElement - %p", this);
    ASSERT(hasTagName(sourceTag));
}

Ref<HTMLSourceElement> HTMLSourceElement::create(const QualifiedName& tagName, Document& document)
{
    auto sourceElement = adoptRef(*new HTMLSourceElement(tagName, document));
    sourceElement->suspendIfNeeded();
    return sourceElement;
}

Ref<HTMLSourceElement> HTMLSourceElement::create(Document& document)
{
    return create(sourceTag, document);
}

Node::InsertedIntoAncestorResult HTMLSourceElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    HTMLElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    RefPtr<Element> parent = parentElement();
    if (parent == &parentOfInsertedTree) {
#if ENABLE(VIDEO)
        if (RefPtr mediaElement = dynamicDowncast<HTMLMediaElement>(*parent))
            mediaElement->sourceWasAdded(*this);
        else
#endif
#if ENABLE(MODEL_ELEMENT)
        if (RefPtr modelElement = dynamicDowncast<HTMLModelElement>(*parent))
            modelElement->sourcesChanged();
        else
#endif
        if (RefPtr pictureElement = dynamicDowncast<HTMLPictureElement>(*parent)) {
            // The new source element only is a relevant mutation if it precedes any img element.
            m_shouldCallSourcesChanged = true;
            for (const Node* node = previousSibling(); node; node = node->previousSibling()) {
                if (is<HTMLImageElement>(*node))
                    m_shouldCallSourcesChanged = false;
            }
            if (m_shouldCallSourcesChanged)
                pictureElement->sourcesChanged();
        }
    }
    return InsertedIntoAncestorResult::Done;
}

void HTMLSourceElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{        
    HTMLElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
    if (!parentNode() && is<Element>(oldParentOfRemovedTree)) {
#if ENABLE(VIDEO)
        if (RefPtr medialElement = dynamicDowncast<HTMLMediaElement>(oldParentOfRemovedTree))
            medialElement->sourceWasRemoved(*this);
        else
#endif
#if ENABLE(MODEL_ELEMENT)
        if (RefPtr model = dynamicDowncast<HTMLModelElement>(oldParentOfRemovedTree))
            model->sourcesChanged();
        else
#endif
        if (m_shouldCallSourcesChanged) {
            downcast<HTMLPictureElement>(oldParentOfRemovedTree).sourcesChanged();
            m_shouldCallSourcesChanged = false;
        }
    }
}

void HTMLSourceElement::didMoveToNewDocument(Document& oldDocument, Document& newDocument)
{
    HTMLElement::didMoveToNewDocument(oldDocument, newDocument);
    ActiveDOMObject::didMoveToNewDocument(newDocument);
}

void HTMLSourceElement::scheduleErrorEvent()
{
    LOG(Media, "HTMLSourceElement::scheduleErrorEvent - %p", this);
    if (m_errorEventCancellationGroup.hasPendingTask())
        return;

    queueCancellableTaskToDispatchEvent(*this, TaskSource::MediaElement, m_errorEventCancellationGroup, Event::create(eventNames().errorEvent, Event::CanBubble::No, Event::IsCancelable::Yes));
}

void HTMLSourceElement::cancelPendingErrorEvent()
{
    LOG(Media, "HTMLSourceElement::cancelPendingErrorEvent - %p", this);
    m_errorEventCancellationGroup.cancel();
}

bool HTMLSourceElement::isURLAttribute(const Attribute& attribute) const
{
    return attribute.name() == srcAttr || HTMLElement::isURLAttribute(attribute);
}

bool HTMLSourceElement::attributeContainsURL(const Attribute& attribute) const
{
    return attribute.name() == srcsetAttr
        || HTMLElement::attributeContainsURL(attribute);
}

void HTMLSourceElement::stop()
{
    cancelPendingErrorEvent();
}

void HTMLSourceElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    HTMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);

    switch (name.nodeName()) {
    case AttributeNames::typeAttr: {
        RefPtr parent = parentElement();
        if (m_shouldCallSourcesChanged && parent)
            downcast<HTMLPictureElement>(*parent).sourcesChanged();
#if ENABLE(MODEL_ELEMENT)
        if (RefPtr parentModelElement = dynamicDowncast<HTMLModelElement>(parent.get()))
            parentModelElement->sourcesChanged();
#endif
        break;
    }
    case AttributeNames::srcsetAttr:
    case AttributeNames::sizesAttr:
    case AttributeNames::mediaAttr: {
        if (name == mediaAttr)
            m_cachedParsedMediaAttribute = std::nullopt;
        RefPtr parent = parentElement();
        if (m_shouldCallSourcesChanged && parent)
            downcast<HTMLPictureElement>(*parent).sourcesChanged();
        break;
    }
    case AttributeNames::widthAttr:
    case AttributeNames::heightAttr:
        if (RefPtr parent = dynamicDowncast<HTMLPictureElement>(parentElement()))
            parent->sourceDimensionAttributesChanged(*this);
        break;
#if ENABLE(MODEL_ELEMENT)
    case AttributeNames::srcAttr:
        if (RefPtr parent = dynamicDowncast<HTMLModelElement>(parentElement()))
            parent->sourcesChanged();
        break;
#endif
    default:
        break;
    }
}

const MQ::MediaQueryList& HTMLSourceElement::parsedMediaAttribute(Document& document) const
{
    if (!m_cachedParsedMediaAttribute) {
        auto& value = attributeWithoutSynchronization(mediaAttr);
        m_cachedParsedMediaAttribute = MQ::MediaQueryParser::parse(value, MediaQueryParserContext { document });
    }
    return m_cachedParsedMediaAttribute.value();
}

Attribute HTMLSourceElement::replaceURLsInAttributeValue(const Attribute& attribute, const UncheckedKeyHashMap<String, String>& replacementURLStrings) const
{
    if (attribute.name() != srcsetAttr)
        return attribute;

    if (replacementURLStrings.isEmpty())
        return attribute;

    return Attribute { srcsetAttr, AtomString { replaceURLsInSrcsetAttribute(*this, StringView(attribute.value()), replacementURLStrings) } };
}

void HTMLSourceElement::addCandidateSubresourceURLs(ListHashSet<URL>& urls) const
{
    getURLsFromSrcsetAttribute(*this, attributeWithoutSynchronization(srcsetAttr), urls);
}

Ref<Element> HTMLSourceElement::cloneElementWithoutAttributesAndChildren(TreeScope& treeScope)
{
    Ref document = treeScope.documentScope();
    Ref clone = create(document);
#if ENABLE(ATTACHMENT_ELEMENT)
    cloneAttachmentAssociatedElementWithoutAttributesAndChildren(clone, document);
#endif
    return clone;
}

void HTMLSourceElement::copyNonAttributePropertiesFromElement(const Element& source)
{
#if ENABLE(ATTACHMENT_ELEMENT)
    auto& sourceElement = downcast<HTMLSourceElement>(source);
    copyAttachmentAssociatedPropertiesFromElement(sourceElement);
#endif
    Element::copyNonAttributePropertiesFromElement(source);
}

}
