/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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
#include "NamedNodeMap.h"

#include "Attr.h"
#include "ElementInlines.h"
#include "HTMLElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace HTMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(NamedNodeMap);

void NamedNodeMap::ref()
{
    m_element->ref();
}

void NamedNodeMap::deref()
{
    m_element->deref();
}

RefPtr<Attr> NamedNodeMap::getNamedItem(const AtomString& name) const
{
    return protectedElement()->getAttributeNode(name);
}

bool NamedNodeMap::isSupportedPropertyName(const AtomString& name) const
{
    return protectedElement()->hasAttribute(name);
}

RefPtr<Attr> NamedNodeMap::getNamedItemNS(const AtomString& namespaceURI, const AtomString& localName) const
{
    return protectedElement()->getAttributeNodeNS(namespaceURI, localName);
}

ExceptionOr<Ref<Attr>> NamedNodeMap::removeNamedItem(const AtomString& name)
{
    Ref element = m_element.get();
    if (!element->hasAttributes())
        return Exception { ExceptionCode::NotFoundError };
    auto index = element->findAttributeIndexByName(name, shouldIgnoreAttributeCase(m_element));
    if (index == ElementData::attributeNotFound)
        return Exception { ExceptionCode::NotFoundError };
    return element->detachAttribute(index);
}

Element& NamedNodeMap::element()
{
    return m_element.get();
}

Ref<Element> NamedNodeMap::protectedElement() const
{
    return m_element.get();
}

Vector<String> NamedNodeMap::supportedPropertyNames() const
{
    Vector<String> names = m_element->getAttributeNames();
    if (is<HTMLElement>(m_element.get()) && m_element->document().isHTMLDocument()) {
        names.removeAllMatching([](String& name) {
            for (auto character : StringView { name }.codeUnits()) {
                if (isASCIIUpper(character))
                    return true;
            }
            return false;
        });
    }
    return names;
}

ExceptionOr<Ref<Attr>> NamedNodeMap::removeNamedItemNS(const AtomString& namespaceURI, const AtomString& localName)
{
    Ref element = m_element.get();
    if (!element->hasAttributes())
        return Exception { ExceptionCode::NotFoundError };
    auto index = element->findAttributeIndexByName(QualifiedName { nullAtom(), localName, namespaceURI });
    if (index == ElementData::attributeNotFound)
        return Exception { ExceptionCode::NotFoundError };
    return element->detachAttribute(index);
}

ExceptionOr<RefPtr<Attr>> NamedNodeMap::setNamedItem(Attr& attr)
{
    return protectedElement()->setAttributeNode(attr);
}

RefPtr<Attr> NamedNodeMap::item(unsigned index) const
{
    if (index >= length())
        return nullptr;
    Ref element = m_element.get();
    return element->ensureAttr(element->attributeAt(index).name());
}

unsigned NamedNodeMap::length() const
{
    if (!m_element->hasAttributes())
        return 0;
    return m_element->attributeCount();
}

} // namespace WebCore
