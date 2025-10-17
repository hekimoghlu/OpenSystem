/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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

#include "AtomHTMLToken.h"
#include "DocumentFragment.h"
#include "Element.h"
#include "Namespace.h"
#include "NodeName.h"
#include "TagName.h"

namespace WebCore {

class HTMLStackItem {
public:
    HTMLStackItem() = default;

    // Normal HTMLElementStack and HTMLFormattingElementList items.
    HTMLStackItem(Ref<Element>&&, AtomHTMLToken&&);
    HTMLStackItem(Ref<Element>&&, Vector<Attribute>&&);

    // Document fragment or element for parsing context.
    explicit HTMLStackItem(Element&);
    explicit HTMLStackItem(DocumentFragment&);

    bool isNull() const { return !m_node; }
    bool isElement() const { return m_node && is<Element>(*m_node); }
    bool isDocumentFragment() const { return m_node && is<DocumentFragment>(*m_node); }

    ContainerNode& node() const { return *m_node; }
    Element& element() const { return downcast<Element>(node()); }
    Ref<Element> protectedElement() const { return element(); }
    Element* elementOrNull() const { return downcast<Element>(m_node.get()); }

    const AtomString& localName() const { return isElement() ? element().localName() : nullAtom(); }
    const AtomString& namespaceURI() const { return isElement() ? element().namespaceURI() : nullAtom(); }

    ElementName elementName() const { return m_elementName; }
    Namespace nodeNamespace() const { return m_namespace; }

    const Vector<Attribute>& attributes() const;
    const Attribute* findAttribute(const QualifiedName& attributeName) const;

    bool matchesHTMLTag(const AtomString&) const;

private:
    ElementName m_elementName = ElementName::Unknown;
    Namespace m_namespace = Namespace::Unknown;
    RefPtr<ContainerNode> m_node;
    Vector<Attribute> m_attributes;
};

bool isInHTMLNamespace(const HTMLStackItem&);
bool isNumberedHeaderElement(const HTMLStackItem&);
bool isSpecialNode(const HTMLStackItem&);

inline HTMLStackItem::HTMLStackItem(Ref<Element>&& element, AtomHTMLToken&& token)
    : m_elementName(element->elementName())
    , m_namespace(element->nodeNamespace())
    , m_node(WTFMove(element))
    , m_attributes(WTFMove(token.attributes()))
{
}

inline HTMLStackItem::HTMLStackItem(Ref<Element>&& element, Vector<Attribute>&& attributes)
    : m_elementName(element->elementName())
    , m_namespace(element->nodeNamespace())
    , m_node(WTFMove(element))
    , m_attributes(WTFMove(attributes))
{
}

inline HTMLStackItem::HTMLStackItem(Element& element)
    : m_elementName(element.elementName())
    , m_namespace(element.nodeNamespace())
    , m_node(&element)
{
}

inline HTMLStackItem::HTMLStackItem(DocumentFragment& fragment)
    : m_node(&fragment)
{
}

inline const Vector<Attribute>& HTMLStackItem::attributes() const
{
    ASSERT(isElement());
    return m_attributes;
}

inline const Attribute* HTMLStackItem::findAttribute(const QualifiedName& attributeName) const
{
    return WebCore::findAttribute(const_cast<Vector<Attribute>&>(attributes()), attributeName);
}

inline bool HTMLStackItem::matchesHTMLTag(const AtomString& name) const
{
    return localName() == name && m_namespace == Namespace::HTML;
}

inline bool isInHTMLNamespace(const HTMLStackItem& item)
{
    // A DocumentFragment takes the place of the document element when parsing
    // fragments and thus should be treated as if it was in the HTML namespace.
    // FIXME: Is this also needed for a ShadowRoot that might be a non-HTML element?
    return item.nodeNamespace() == Namespace::HTML || item.isDocumentFragment();
}

inline bool isNumberedHeaderElement(const HTMLStackItem& item)
{
    using namespace ElementNames;

    switch (item.elementName()) {
    case HTML::h1:
    case HTML::h2:
    case HTML::h3:
    case HTML::h4:
    case HTML::h5:
    case HTML::h6:
        return true;
    default:
        return false;
    }
}

// https://html.spec.whatwg.org/multipage/parsing.html#special
inline bool isSpecialNode(const HTMLStackItem& item)
{
    using namespace ElementNames;

    if (item.isDocumentFragment())
        return true;

    switch (item.elementName()) {
    case HTML::address:
    case HTML::applet:
    case HTML::area:
    case HTML::article:
    case HTML::aside:
    case HTML::base:
    case HTML::basefont:
    case HTML::bgsound:
    case HTML::blockquote:
    case HTML::body:
    case HTML::br:
    case HTML::button:
    case HTML::caption:
    case HTML::center:
    case HTML::col:
    case HTML::colgroup:
    case HTML::dd:
    case HTML::details:
    case HTML::dir:
    case HTML::div:
    case HTML::dl:
    case HTML::dt:
    case HTML::embed:
    case HTML::fieldset:
    case HTML::figcaption:
    case HTML::figure:
    case HTML::footer:
    case HTML::form:
    case HTML::frame:
    case HTML::frameset:
    case HTML::h1:
    case HTML::h2:
    case HTML::h3:
    case HTML::h4:
    case HTML::h5:
    case HTML::h6:
    case HTML::head:
    case HTML::header:
    case HTML::hgroup:
    case HTML::hr:
    case HTML::html:
    case HTML::iframe:
    case HTML::img:
    case HTML::input:
    case HTML::li:
    case HTML::link:
    case HTML::listing:
    case HTML::main:
    case HTML::marquee:
    case HTML::menu:
    case HTML::meta:
    case HTML::nav:
    case HTML::noembed:
    case HTML::noframes:
    case HTML::noscript:
    case HTML::object:
    case HTML::ol:
    case HTML::p:
    case HTML::param:
    case HTML::plaintext:
    case HTML::pre:
    case HTML::script:
    case HTML::section:
    case HTML::select:
    case HTML::style:
    case HTML::summary:
    case HTML::table:
    case HTML::tbody:
    case HTML::td:
    case HTML::template_:
    case HTML::textarea:
    case HTML::tfoot:
    case HTML::th:
    case HTML::thead:
    case HTML::tr:
    case HTML::ul:
    case HTML::wbr:
    case HTML::xmp:
    case MathML::annotation_xml:
    case MathML::mi:
    case MathML::mo:
    case MathML::mn:
    case MathML::ms:
    case MathML::mtext:
    case SVG::desc:
    case SVG::foreignObject:
    case SVG::title:
        return true;
    default:
        return false;
    }
}

} // namespace WebCore
