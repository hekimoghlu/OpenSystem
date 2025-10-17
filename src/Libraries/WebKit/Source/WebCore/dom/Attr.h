/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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

#include "Node.h"
#include "QualifiedName.h"

namespace WebCore {

class Attribute;
class CSSStyleDeclaration;
class MutableStyleProperties;

class Attr final : public Node {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Attr);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(Attr);
public:
    static Ref<Attr> create(Element&, const QualifiedName&);
    static Ref<Attr> create(Document&, const QualifiedName&, const AtomString& value);
    virtual ~Attr();

    String name() const { return qualifiedName().toString(); }
    bool specified() const { return true; }
    Element* ownerElement() const { return m_element.get(); }

    WEBCORE_EXPORT AtomString value() const;
    WEBCORE_EXPORT ExceptionOr<void> setValue(const AtomString&);

    const QualifiedName& qualifiedName() const { return m_name; }

    WEBCORE_EXPORT CSSStyleDeclaration* style();

    void attachToElement(Element&);
    void detachFromElementWithValue(const AtomString&);

    const AtomString& namespaceURI() const final { return m_name.namespaceURI(); }
    const AtomString& localName() const final { return m_name.localName(); }
    const AtomString& prefix() const final { return m_name.prefix(); }

private:
    Attr(Element&, const QualifiedName&);
    Attr(Document&, const QualifiedName&, const AtomString& value);

    String nodeName() const final { return name(); }

    String nodeValue() const final { return value(); }
    ExceptionOr<void> setNodeValue(const String&) final;

    ExceptionOr<void> setPrefix(const AtomString&) final;

    Ref<Node> cloneNodeInternal(TreeScope&, CloningOperation) final;

    bool isAttributeNode() const final { return true; }

    void parentOrShadowHostNode() const = delete; // Call parentNode() instead.

    // Attr wraps either an element/name, or a name/value pair (when it's a standalone Node.)
    // Note that m_name is always set, but m_element/m_standaloneValue may be null.
    WeakPtr<Element, WeakPtrImplWithEventTargetData> m_element;
    QualifiedName m_name;
    AtomString m_standaloneValue;

    RefPtr<MutableStyleProperties> m_style;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::Attr)
    static bool isType(const WebCore::Node& node) { return node.isAttributeNode(); }
SPECIALIZE_TYPE_TRAITS_END()
