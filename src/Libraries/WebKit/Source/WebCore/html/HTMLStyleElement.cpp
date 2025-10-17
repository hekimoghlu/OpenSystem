/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#include "HTMLStyleElement.h"

#include "CSSRule.h"
#include "CachedResource.h"
#include "DOMTokenList.h"
#include "Document.h"
#include "Event.h"
#include "EventNames.h"
#include "EventSender.h"
#include "HTMLNames.h"
#include "MediaQueryParser.h"
#include "MediaQueryParserContext.h"
#include "NodeName.h"
#include "Page.h"
#include "ScriptableDocumentParser.h"
#include "ShadowRoot.h"
#include "StyleScope.h"
#include "StyleSheetContents.h"
#include "TextNodeTraversal.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLStyleElement);

using namespace HTMLNames;

static StyleEventSender& styleLoadEventSender()
{
    static NeverDestroyed<StyleEventSender> sharedLoadEventSender;
    return sharedLoadEventSender;
}

inline HTMLStyleElement::HTMLStyleElement(const QualifiedName& tagName, Document& document, bool createdByParser)
    : HTMLElement(tagName, document)
    , m_styleSheetOwner(document, createdByParser)
{
    ASSERT(hasTagName(styleTag));
}

HTMLStyleElement::~HTMLStyleElement()
{
    m_styleSheetOwner.clearDocumentData(*this);

    styleLoadEventSender().cancelEvent(*this);
}

Ref<HTMLStyleElement> HTMLStyleElement::create(const QualifiedName& tagName, Document& document, bool createdByParser)
{
    return adoptRef(*new HTMLStyleElement(tagName, document, createdByParser));
}

Ref<HTMLStyleElement> HTMLStyleElement::create(Document& document)
{
    return adoptRef(*new HTMLStyleElement(styleTag, document, false));
}

void HTMLStyleElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::titleAttr:
        if (sheet() && !isInShadowTree())
            sheet()->setTitle(newValue);
        break;
    case AttributeNames::mediaAttr:
        m_styleSheetOwner.setMedia(newValue);
        if (sheet()) {
            sheet()->setMediaQueries(MQ::MediaQueryParser::parse(newValue, MediaQueryParserContext(document())));
            if (auto* scope = m_styleSheetOwner.styleScope())
                scope->didChangeStyleSheetContents();
        } else
            m_styleSheetOwner.childrenChanged(*this);
        break;
    case AttributeNames::typeAttr:
        m_styleSheetOwner.setContentType(newValue);
        m_styleSheetOwner.childrenChanged(*this);
        if (auto* scope = m_styleSheetOwner.styleScope())
            scope->didChangeStyleSheetContents();
        break;
    case AttributeNames::blockingAttr:
        if (m_blockingList)
            m_blockingList->associatedAttributeValueChanged();
        break;
    default:
        HTMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
        break;
    }
}

// https://html.spec.whatwg.org/multipage/semantics.html#dom-style-blocking
// FIXME: Bug 88869 - This isn't currently connected to anything, because style elements are
// parser blocking, even when script inserted.
DOMTokenList& HTMLStyleElement::blocking()
{
    if (!m_blockingList) {
        m_blockingList = makeUniqueWithoutRefCountedCheck<DOMTokenList>(*this, HTMLNames::blockingAttr, [](Document&, StringView token) {
            if (equalLettersIgnoringASCIICase(token, "render"_s))
                return true;
            return false;
        });
    }
    return *m_blockingList;
}

void HTMLStyleElement::finishParsingChildren()
{
    m_styleSheetOwner.finishParsingChildren(*this);
    HTMLElement::finishParsingChildren();
}

Node::InsertedIntoAncestorResult HTMLStyleElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    auto result = HTMLElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (insertionType.connectedToDocument)
        m_styleSheetOwner.insertedIntoDocument(*this);
    return result;
}

void HTMLStyleElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    HTMLElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
    if (removalType.disconnectedFromDocument)
        m_styleSheetOwner.removedFromDocument(*this);
}

void HTMLStyleElement::childrenChanged(const ChildChange& change)
{
    HTMLElement::childrenChanged(change);
    m_styleSheetOwner.childrenChanged(*this);
}

void HTMLStyleElement::dispatchPendingLoadEvents(Page* page)
{
    styleLoadEventSender().dispatchPendingEvents(page);
}

void HTMLStyleElement::dispatchPendingEvent(StyleEventSender* eventSender, const AtomString& eventType)
{
    ASSERT_UNUSED(eventSender, eventSender == &styleLoadEventSender());
    dispatchEvent(Event::create(eventType, Event::CanBubble::No, Event::IsCancelable::No));
}

void HTMLStyleElement::notifyLoadedSheetAndAllCriticalSubresources(bool errorOccurred)
{
    m_loadedSheet = !errorOccurred;
    styleLoadEventSender().dispatchEventSoon(*this, m_loadedSheet ? eventNames().loadEvent : eventNames().errorEvent);
}

void HTMLStyleElement::addSubresourceAttributeURLs(ListHashSet<URL>& urls) const
{
    HTMLElement::addSubresourceAttributeURLs(urls);

    if (RefPtr styleSheet = this->sheet()) {
        styleSheet->contents().traverseSubresources([&] (auto& resource) {
            urls.add(resource.url());
            return false;
        });
    }
}

bool HTMLStyleElement::disabled() const
{
    if (!sheet())
        return false;

    return sheet()->disabled();
}

void HTMLStyleElement::setDisabled(bool setDisabled)
{
    if (CSSStyleSheet* styleSheet = sheet())
        styleSheet->setDisabled(setDisabled);
}

}
