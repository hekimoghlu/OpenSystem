/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
#include "Attr.h"

#include "AttributeChangeInvalidation.h"
#include "CommonAtomStrings.h"
#include "Document.h"
#include "ElementInlines.h"
#include "Event.h"
#include "HTMLNames.h"
#include "MutableStyleProperties.h"
#include "ScopedEventQueue.h"
#include "StyledElement.h"
#include "TextNodeTraversal.h"
#include "XMLNSNames.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Attr);

using namespace HTMLNames;

Attr::Attr(Element& element, const QualifiedName& name)
    : Node(element.document(), ATTRIBUTE_NODE, { })
    , m_element(element)
    , m_name(name)
{
}

Attr::Attr(Document& document, const QualifiedName& name, const AtomString& standaloneValue)
    : Node(document, ATTRIBUTE_NODE, { })
    , m_name(name)
    , m_standaloneValue(standaloneValue)
{
}

Ref<Attr> Attr::create(Element& element, const QualifiedName& name)
{
    return adoptRef(*new Attr(element, name));
}

Ref<Attr> Attr::create(Document& document, const QualifiedName& name, const AtomString& value)
{
    return adoptRef(*new Attr(document, name, value));
}

Attr::~Attr()
{
    ASSERT_WITH_SECURITY_IMPLICATION(!isInShadowTree());
    ASSERT_WITH_SECURITY_IMPLICATION(treeScope().rootNode().isDocumentNode());

    // Unable to protect the document here as it may have started destruction.
    willBeDeletedFrom(document());
}

ExceptionOr<void> Attr::setPrefix(const AtomString& prefix)
{
    auto result = checkSetPrefix(prefix);
    if (result.hasException())
        return result.releaseException();

    if ((prefix == xmlnsAtom() && namespaceURI() != XMLNSNames::xmlnsNamespaceURI) || qualifiedName() == xmlnsAtom())
        return Exception { ExceptionCode::NamespaceError };

    const AtomString& newPrefix = prefix.isEmpty() ? nullAtom() : prefix;
    if (RefPtr element = m_element.get())
        element->ensureUniqueElementData().findAttributeByName(qualifiedName())->setPrefix(newPrefix);

    m_name.setPrefix(newPrefix);

    return { };
}

ExceptionOr<void> Attr::setValue(const AtomString& value)
{
    if (RefPtr element = m_element.get())
        return element->setAttribute(qualifiedName(), value, true);
    else
        m_standaloneValue = value;

    return { };
}

ExceptionOr<void> Attr::setNodeValue(const String& value)
{
    return setValue(value.isNull() ? emptyAtom() : AtomString(value));
}

Ref<Node> Attr::cloneNodeInternal(TreeScope& treeScope, CloningOperation)
{
    return adoptRef(*new Attr(treeScope.documentScope(), qualifiedName(), value()));
}

CSSStyleDeclaration* Attr::style()
{
    // This is not part of the DOM API, and therefore not available to webpages. However, WebKit SPI
    // lets clients use this via the Objective-C and JavaScript bindings.
    RefPtr styledElement = dynamicDowncast<StyledElement>(m_element.get());
    if (!styledElement)
        return nullptr;
    Ref style = MutableStyleProperties::create();
    m_style = style.copyRef();
    styledElement->collectPresentationalHintsForAttribute(qualifiedName(), value(), style);
    return &style->ensureCSSStyleDeclaration();
}

AtomString Attr::value() const
{
    if (RefPtr element = m_element.get())
        return element->getAttributeForBindings(qualifiedName());
    return m_standaloneValue;
}

void Attr::detachFromElementWithValue(const AtomString& value)
{
    ASSERT(m_element);
    ASSERT(m_standaloneValue.isNull());
    m_standaloneValue = value;
    m_element = nullptr;
    setTreeScopeRecursively(Ref<Document> { document() });
}

void Attr::attachToElement(Element& element)
{
    ASSERT(!m_element);
    m_element = &element;
    m_standaloneValue = nullAtom();
    setTreeScopeRecursively(element.treeScope());
}

}
