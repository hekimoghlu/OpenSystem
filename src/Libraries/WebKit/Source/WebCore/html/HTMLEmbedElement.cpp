/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
#include "HTMLEmbedElement.h"

#include "CSSPropertyNames.h"
#include "ElementAncestorIteratorInlines.h"
#include "FrameLoader.h"
#include "HTMLImageLoader.h"
#include "HTMLNames.h"
#include "HTMLObjectElement.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "NodeName.h"
#include "PluginDocument.h"
#include "RenderEmbeddedObject.h"
#include "RenderWidget.h"
#include "Settings.h"
#include "SubframeLoader.h"
#include <wtf/Ref.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLEmbedElement);

using namespace HTMLNames;

inline HTMLEmbedElement::HTMLEmbedElement(const QualifiedName& tagName, Document& document)
    : HTMLPlugInImageElement(tagName, document)
{
    ASSERT(hasTagName(embedTag));
}

Ref<HTMLEmbedElement> HTMLEmbedElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLEmbedElement(tagName, document));
}

Ref<HTMLEmbedElement> HTMLEmbedElement::create(Document& document)
{
    return create(embedTag, document);
}

static inline RenderWidget* findWidgetRenderer(const Node* node)
{
    if (!node->renderer())
        node = ancestorsOfType<HTMLObjectElement>(*node).first();
    if (node)
        return dynamicDowncast<RenderWidget>(node->renderer());
    return nullptr;
}

RenderWidget* HTMLEmbedElement::renderWidgetLoadingPlugin() const
{
    RenderWidget* widget = HTMLPlugInImageElement::renderWidgetLoadingPlugin();

    return widget ? widget : findWidgetRenderer(this);
}

void HTMLEmbedElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    if (name == hiddenAttr) {
        ASSERT(!value.isNull());
        addPropertyToPresentationalHintStyle(style, CSSPropertyWidth, 0, CSSUnitType::CSS_PX);
        addPropertyToPresentationalHintStyle(style, CSSPropertyHeight, 0, CSSUnitType::CSS_PX);
    } else
        HTMLPlugInImageElement::collectPresentationalHintsForAttribute(name, value, style);
}

static bool hasTypeOrSrc(const HTMLEmbedElement& embed)
{
    return embed.hasAttributeWithoutSynchronization(typeAttr) || embed.hasAttributeWithoutSynchronization(srcAttr);
}

void HTMLEmbedElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    HTMLPlugInImageElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
    switch (name.nodeName()) {
    case AttributeNames::typeAttr:
        m_serviceType = newValue.string().left(newValue.find(';')).convertToASCIILowercase();
        // FIXME: The only difference between this and HTMLObjectElement's corresponding
        // code is that HTMLObjectElement does setNeedsWidgetUpdate(true). Consider moving
        // this up to the HTMLPlugInImageElement to be shared.
        if (renderer() && !hasTypeOrSrc(*this))
            invalidateStyle();
        break;
    case AttributeNames::codeAttr:
        // FIXME: trimming whitespace is probably redundant with the URL parser
        m_url = newValue.string().trim(isASCIIWhitespace);
        // FIXME: Why no call to updateImageLoaderWithNewURLSoon?
        // FIXME: If both code and src attributes are specified, last one parsed/changed wins. That can't be right!
        break;
    case AttributeNames::srcAttr:
        // FIXME: trimming whitespace is probably redundant with the URL parser
        m_url = newValue.string().trim(isASCIIWhitespace);
        updateImageLoaderWithNewURLSoon();
        if (renderer() && !hasTypeOrSrc(*this))
            invalidateStyle();
        // FIXME: If both code and src attributes are specified, last one parsed/changed wins. That can't be right!
        break;
    default:
        break;
    }
}

void HTMLEmbedElement::parametersForPlugin(Vector<AtomString>& paramNames, Vector<AtomString>& paramValues)
{
    if (!hasAttributes())
        return;

    for (auto& attribute : attributes()) {
        paramNames.append(attribute.localName());
        paramValues.append(attribute.value());
    }
}

// FIXME: This should be unified with HTMLObjectElement::updateWidget and
// moved down into HTMLPluginImageElement.cpp
void HTMLEmbedElement::updateWidget(CreatePlugins createPlugins)
{
    ASSERT(!renderEmbeddedObject()->isPluginUnavailable());
    ASSERT(needsWidgetUpdate());

    if (m_url.isEmpty() && m_serviceType.isEmpty()) {
        setNeedsWidgetUpdate(false);
        return;
    }

    // Note these pass m_url and m_serviceType to allow better code sharing with
    // <object> which modifies url and serviceType before calling these.
    if (!canLoadURL(m_url)) {
        setNeedsWidgetUpdate(false);
        return;
    }

    // FIXME: It's unfortunate that we have this special case here.
    // See http://trac.webkit.org/changeset/25128 and the plugins/netscape-plugin-setwindow-size.html test.
    if (createPlugins == CreatePlugins::No && wouldLoadAsPlugIn(m_url, m_serviceType))
        return;

    setNeedsWidgetUpdate(false);

    // FIXME: These should be joined into a PluginParameters class.
    Vector<AtomString> paramNames;
    Vector<AtomString> paramValues;
    parametersForPlugin(paramNames, paramValues);

    Ref<HTMLEmbedElement> protectedThis(*this); // Loading the plugin might remove us from the document.
    if (!renderer()) // Do not load the plugin if beforeload removed this element or its renderer.
        return;

    // Dispatching a beforeLoad event could have executed code that changed the document.
    // Make sure the URL is still safe to load.
    if (!canLoadURL(m_url))
        return;

    // FIXME: beforeLoad could have detached the renderer!  Just like in the <object> case above.
    requestObject(m_url, m_serviceType, paramNames, paramValues);
}

bool HTMLEmbedElement::rendererIsNeeded(const RenderStyle& style)
{
    if (!hasTypeOrSrc(*this))
        return false;

    if (isImageType())
        return HTMLPlugInImageElement::rendererIsNeeded(style);

    // If my parent is an <object> and is not set to use fallback content, I
    // should be ignored and not get a renderer.
    if (RefPtr parentObject = dynamicDowncast<HTMLObjectElement>(parentNode())) {
        if (!parentObject->renderer())
            return false;
        if (!parentObject->useFallbackContent()) {
            ASSERT(!parentObject->renderer()->isRenderEmbeddedObject());
            return false;
        }
    }

    return HTMLPlugInImageElement::rendererIsNeeded(style);
}

bool HTMLEmbedElement::isURLAttribute(const Attribute& attribute) const
{
    return attribute.name() == srcAttr || HTMLPlugInImageElement::isURLAttribute(attribute);
}

const AtomString& HTMLEmbedElement::imageSourceURL() const
{
    return attributeWithoutSynchronization(srcAttr);
}

void HTMLEmbedElement::addSubresourceAttributeURLs(ListHashSet<URL>& urls) const
{
    HTMLPlugInImageElement::addSubresourceAttributeURLs(urls);

    addSubresourceURL(urls, document().completeURL(attributeWithoutSynchronization(srcAttr)));
}

}
