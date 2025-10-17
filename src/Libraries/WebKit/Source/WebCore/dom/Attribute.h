/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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

#include "CommonAtomStrings.h"
#include "QualifiedName.h"
#include <wtf/Hasher.h>

namespace WebCore {

// This has no counterpart in DOM.
// It is an internal representation of the node value of an Attr.
// The actual Attr with its value as a Text child is allocated only if needed.
class Attribute {
public:
    Attribute(const QualifiedName& name, const AtomString& value)
        : m_name(name)
        , m_value(value)
    {
    }

    // NOTE: The references returned by these functions are only valid for as long
    // as the Attribute stays in place. For example, calling a function that mutates
    // an Element's internal attribute storage may invalidate them.
    const AtomString& value() const { return m_value; }
    static constexpr ptrdiff_t valueMemoryOffset() { return OBJECT_OFFSETOF(Attribute, m_value); }
    const AtomString& prefix() const { return m_name.prefix(); }
    const AtomString& localName() const { return m_name.localName(); }
    const AtomString& localNameLowercase() const { return m_name.localNameLowercase(); }
    const AtomString& namespaceURI() const { return m_name.namespaceURI(); }

    const QualifiedName& name() const { return m_name; }
    static constexpr ptrdiff_t nameMemoryOffset() { return OBJECT_OFFSETOF(Attribute, m_name); }

    bool isEmpty() const { return m_value.isEmpty(); }
    static bool nameMatchesFilter(const QualifiedName&, const AtomString& filterPrefix, const AtomString& filterLocalName, const AtomString& filterNamespaceURI);
    bool matches(const AtomString& prefix, const AtomString& localName, const AtomString& namespaceURI) const;

    void setValue(const AtomString& value) { m_value = value; }
    void setPrefix(const AtomString& prefix) { m_name.setPrefix(prefix); }

    // Note: This API is only for HTMLTreeBuilder.  It is not safe to change the
    // name of an attribute once parseAttribute has been called as DOM
    // elements may have placed the Attribute in a hash by name.
    void parserSetName(const QualifiedName& name) { m_name = name; }

#if COMPILER(MSVC)
    // NOTE: This constructor is not actually implemented, it's just defined so MSVC
    // will let us use a zero-length array of Attributes.
    Attribute();
#endif

private:
    QualifiedName m_name;
    AtomString m_value;
};

inline void add(Hasher& hasher, const Attribute& attribute)
{
    add(hasher, attribute.name(), attribute.value());
}

inline bool Attribute::nameMatchesFilter(const QualifiedName& name, const AtomString& filterPrefix, const AtomString& filterLocalName, const AtomString& filterNamespaceURI)
{
    if (filterLocalName != name.localName())
        return false;
    return filterPrefix == starAtom() || filterNamespaceURI == name.namespaceURI();
}

inline bool Attribute::matches(const AtomString& prefix, const AtomString& localName, const AtomString& namespaceURI) const
{
    return nameMatchesFilter(m_name, prefix, localName, namespaceURI);
}

} // namespace WebCore

namespace WTF {

template<>
struct VectorTraits<WebCore::Attribute> : SimpleClassVectorTraits { };

} // namespace WTF
