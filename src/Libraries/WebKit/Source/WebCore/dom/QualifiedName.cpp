/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
#include "QualifiedName.h"

#include "CommonAtomStrings.h"
#include "Namespace.h"
#include "NodeName.h"
#include "QualifiedNameCache.h"
#include "ThreadGlobalData.h"
#include <wtf/Assertions.h>

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(QualifiedName);
DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(QualifiedNameQualifiedNameImpl);

QualifiedName::QualifiedNameImpl::QualifiedNameImpl(const AtomString& prefix, const AtomString& localName, const AtomString& namespaceURI)
    : m_namespace(Namespace::Unknown)
    , m_nodeName(NodeName::Unknown)
    , m_prefix(prefix)
    , m_localName(localName)
    , m_namespaceURI(namespaceURI)
{
    ASSERT(!namespaceURI.isEmpty() || namespaceURI.isNull());
}

static QualifiedNameComponents makeComponents(const AtomString& prefix, const AtomString& localName, const AtomString& namespaceURI)
{
    return { prefix.impl(), localName.impl(), namespaceURI.isEmpty() ? nullptr : namespaceURI.impl() };
}

QualifiedName::QualifiedName(const AtomString& prefix, const AtomString& localName, const AtomString& namespaceURI)
    : m_impl(threadGlobalData().qualifiedNameCache().getOrCreate(makeComponents(prefix, localName, namespaceURI)))
{
}

QualifiedName::QualifiedName(const AtomString& prefix, const AtomString& localName, const AtomString& namespaceURI, Namespace nodeNamespace, NodeName nodeName)
    : m_impl(threadGlobalData().qualifiedNameCache().getOrCreate(makeComponents(prefix, localName, namespaceURI), nodeNamespace, nodeName))
{
}

QualifiedName::QualifiedNameImpl::~QualifiedNameImpl()
{
    threadGlobalData().qualifiedNameCache().remove(*this);
}

// Global init routines
LazyNeverDestroyed<const QualifiedName> anyName;
LazyNeverDestroyed<const QualifiedName> nullName;

void QualifiedName::init()
{
    static bool initialized = false;
    if (initialized)
        return;

    anyName.construct(nullAtom(), starAtom(), starAtom(), Namespace::Unknown, NodeName::Unknown);
    nullName.construct(nullAtom(), nullAtom(), nullAtom(), Namespace::None, NodeName::Unknown);
    initialized = true;
}

const AtomString& QualifiedName::localNameUppercase() const
{
    if (!m_impl->m_localNameUpper)
        m_impl->m_localNameUpper = m_impl->m_localName.convertToASCIIUppercase();
    return m_impl->m_localNameUpper;
}

unsigned QualifiedName::QualifiedNameImpl::computeHash() const
{
    QualifiedNameComponents components = { m_prefix.impl(), m_localName.impl(), m_namespaceURI.impl() };
    return WTF::computeHash(components);
}

}
