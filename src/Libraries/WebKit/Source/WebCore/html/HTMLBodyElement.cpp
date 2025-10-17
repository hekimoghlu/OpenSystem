/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#include "HTMLBodyElement.h"

#include "CSSImageValue.h"
#include "CSSParser.h"
#include "CSSValueKeywords.h"
#include "DOMWrapperWorld.h"
#include "ElementInlines.h"
#include "EventNames.h"
#include "HTMLFrameElement.h"
#include "HTMLIFrameElement.h"
#include "HTMLNames.h"
#include "JSHTMLBodyElement.h"
#include "LocalDOMWindow.h"
#include "MutableStyleProperties.h"
#include "NodeName.h"
#include "ResourceLoaderOptions.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLBodyElement);

using namespace HTMLNames;

HTMLBodyElement::HTMLBodyElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(bodyTag));
}

Ref<HTMLBodyElement> HTMLBodyElement::create(Document& document)
{
    return adoptRef(*new HTMLBodyElement(bodyTag, document));
}

Ref<HTMLBodyElement> HTMLBodyElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLBodyElement(tagName, document));
}

HTMLBodyElement::~HTMLBodyElement() = default;

bool HTMLBodyElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    switch (name.nodeName()) {
    case AttributeNames::backgroundAttr:
    case AttributeNames::marginwidthAttr:
    case AttributeNames::leftmarginAttr:
    case AttributeNames::marginheightAttr:
    case AttributeNames::topmarginAttr:
    case AttributeNames::bgcolorAttr:
    case AttributeNames::textAttr:
        return true;
    default:
        break;
    }
    return HTMLElement::hasPresentationalHintsForAttribute(name);
}

void HTMLBodyElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    switch (name.nodeName()) {
    case AttributeNames::backgroundAttr: {
        auto url = value.string().trim(isASCIIWhitespace);
        if (!url.isEmpty()) {
            auto imageValue = CSSImageValue::create(document().completeURL(url), LoadedFromOpaqueSource::No, localName());
            style.setProperty(CSSProperty(CSSPropertyBackgroundImage, WTFMove(imageValue)));
        }
        break;
    }
    case AttributeNames::marginwidthAttr:
    case AttributeNames::leftmarginAttr:
        addHTMLLengthToStyle(style, CSSPropertyMarginRight, value);
        addHTMLLengthToStyle(style, CSSPropertyMarginLeft, value);
        break;
    case AttributeNames::marginheightAttr:
    case AttributeNames::topmarginAttr:
        addHTMLLengthToStyle(style, CSSPropertyMarginBottom, value);
        addHTMLLengthToStyle(style, CSSPropertyMarginTop, value);
        break;
    case AttributeNames::bgcolorAttr:
        addHTMLColorToStyle(style, CSSPropertyBackgroundColor, value);
        break;
    case AttributeNames::textAttr:
        addHTMLColorToStyle(style, CSSPropertyColor, value);
        break;
    default:
        HTMLElement::collectPresentationalHintsForAttribute(name, value, style);
        break;
    }
}

const AtomString& HTMLBodyElement::eventNameForWindowEventHandlerAttribute(const QualifiedName& attributeName)
{
    static NeverDestroyed map = [] {
        EventHandlerNameMap map;
        JSHTMLBodyElement::forEachWindowEventHandlerContentAttribute([&] (const AtomString& attributeName, const AtomString& eventName) {
            map.add(attributeName.impl(), eventName);
        });
        return map;
    }();
    return eventNameForEventHandlerAttribute(attributeName, map);
}

void HTMLBodyElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::vlinkAttr:
        if (auto parsedColor = parseLegacyColorValue(newValue))
            document().setVisitedLinkColor(*parsedColor);
        else
            document().resetVisitedLinkColor();
        invalidateStyleForSubtree();
        return;
    case AttributeNames::alinkAttr:
        if (auto parsedColor = parseLegacyColorValue(newValue))
            document().setActiveLinkColor(*parsedColor);
        else
            document().resetActiveLinkColor();
        invalidateStyleForSubtree();
        return;
    case AttributeNames::linkAttr:
        if (auto parsedColor = parseLegacyColorValue(newValue))
            document().setLinkColor(*parsedColor);
        else
            document().resetLinkColor();
        invalidateStyleForSubtree();
        return;
    case AttributeNames::onselectionchangeAttr:
        // FIXME: Emit "selectionchange" event at <input> / <textarea> elements and remove this special-case.
        // https://bugs.webkit.org/show_bug.cgi?id=234348
        document().setAttributeEventListener(eventNames().selectionchangeEvent, name, newValue, mainThreadNormalWorld());
        return;
    default:
        break;
    }

    if (auto& eventName = eventNameForWindowEventHandlerAttribute(name); !eventName.isNull()) {
        document().setWindowAttributeEventListener(eventName, name, newValue, mainThreadNormalWorld());
        return;
    }

    HTMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

Node::InsertedIntoAncestorResult HTMLBodyElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    HTMLElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (!insertionType.connectedToDocument)
        return InsertedIntoAncestorResult::Done;
    if (!is<HTMLFrameElementBase>(document().ownerElement()))
        return InsertedIntoAncestorResult::Done;
    return InsertedIntoAncestorResult::NeedsPostInsertionCallback;
}

void HTMLBodyElement::didFinishInsertingNode()
{
    // A DOM mutation could have happened in between the call to insertedIntoAncestor() and the
    // call to didFinishInsertingNode().
    if (!is<HTMLFrameElementBase>(document().ownerElement()))
        return;

    Ref ownerElement = *document().ownerElement();

    // FIXME: It's surprising this is web compatible since it means marginwidth and marginheight attributes
    // appear or get overwritten on body elements of a document embedded through <iframe> or <frame>.
    // Better to find a way to do addHTMLLengthToStyle based on the attributes from the frame element,
    // without modifying the body element's attributes. Could also add code so we can respond to updates
    // to the frame element attributes.
    auto marginWidth = ownerElement->attributeWithoutSynchronization(marginwidthAttr);
    if (!marginWidth.isNull())
        setAttributeWithoutSynchronization(marginwidthAttr, marginWidth);
    auto marginHeight = ownerElement->attributeWithoutSynchronization(marginheightAttr);
    if (!marginHeight.isNull())
        setAttributeWithoutSynchronization(marginheightAttr, marginHeight);
}

bool HTMLBodyElement::isURLAttribute(const Attribute& attribute) const
{
    return attribute.name() == backgroundAttr || HTMLElement::isURLAttribute(attribute);
}

bool HTMLBodyElement::supportsFocus() const
{
    return hasEditableStyle() || HTMLElement::supportsFocus();
}

void HTMLBodyElement::addSubresourceAttributeURLs(ListHashSet<URL>& urls) const
{
    HTMLElement::addSubresourceAttributeURLs(urls);

    addSubresourceURL(urls, document().completeURL(attributeWithoutSynchronization(backgroundAttr)));
}

} // namespace WebCore
