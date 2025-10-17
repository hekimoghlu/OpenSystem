/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
#include "SVGCursorElement.h"

#include "Document.h"
#include "SVGNames.h"
#include "SVGStringList.h"
#include "StyleCursorImage.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGCursorElement);

inline SVGCursorElement::SVGCursorElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
    , SVGTests(this)
    , SVGURIReference(this)
{
    ASSERT(hasTagName(SVGNames::cursorTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::xAttr, &SVGCursorElement::m_x>();
        PropertyRegistry::registerProperty<SVGNames::yAttr, &SVGCursorElement::m_y>();
    });
}

Ref<SVGCursorElement> SVGCursorElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGCursorElement(tagName, document));
}

SVGCursorElement::~SVGCursorElement()
{
    for (auto& client : m_clients)
        client.cursorElementRemoved(*this);
}

void SVGCursorElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    SVGParsingError parseError = NoError;

    if (name == SVGNames::xAttr)
        Ref { m_x }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Width, newValue, parseError));
    else if (name == SVGNames::yAttr)
        Ref { m_y }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Height, newValue, parseError));

    reportAttributeParsingError(parseError, name, newValue);

    SVGURIReference::parseAttribute(name, newValue);
    SVGTests::parseAttribute(name, newValue);
    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGCursorElement::addClient(StyleCursorImage& value)
{
    m_clients.add(value);
}

void SVGCursorElement::removeClient(StyleCursorImage& value)
{
    m_clients.remove(value);
}

void SVGCursorElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        for (auto& client : m_clients)
            client.cursorElementChanged(*this);
        return;
    }

    SVGElement::svgAttributeChanged(attrName);
}

void SVGCursorElement::addSubresourceAttributeURLs(ListHashSet<URL>& urls) const
{
    SVGElement::addSubresourceAttributeURLs(urls);

    addSubresourceURL(urls, document().completeURL(href()));
}

}
