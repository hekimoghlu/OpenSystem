/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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
#include "QualifiedNameCache.h"

#include "Namespace.h"
#include "NodeName.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(QualifiedNameCache);

struct QNameComponentsTranslator {
    static unsigned hash(const QualifiedNameComponents& components)
    {
        return computeHash(components);
    }

    static bool equal(QualifiedName::QualifiedNameImpl* name, const QualifiedNameComponents& c)
    {
        return c.m_prefix == name->m_prefix.impl() && c.m_localName == name->m_localName.impl() && c.m_namespaceURI == name->m_namespaceURI.impl();
    }

    static void translate(QualifiedName::QualifiedNameImpl*& location, const QualifiedNameComponents& components, unsigned)
    {
        location = &QualifiedName::QualifiedNameImpl::create(components.m_prefix, components.m_localName, components.m_namespaceURI).leakRef();
    }
};

static void updateImplWithNamespaceAndElementName(QualifiedName::QualifiedNameImpl& impl, Namespace nodeNamespace, NodeName nodeName)
{
    impl.m_namespace = nodeNamespace;
    impl.m_nodeName = nodeName;
    bool needsLowercasing = nodeNamespace != Namespace::HTML || nodeName == NodeName::Unknown;
    impl.m_localNameLower = needsLowercasing ? impl.m_localName.convertToASCIILowercase() : impl.m_localName;
}

Ref<QualifiedName::QualifiedNameImpl> QualifiedNameCache::getOrCreate(const QualifiedNameComponents& components)
{
    auto addResult = m_cache.add<QNameComponentsTranslator>(components);
    auto& impl = **addResult.iterator;

    if (addResult.isNewEntry) {
        auto nodeNamespace = findNamespace(components.m_namespaceURI);
        auto nodeName = findNodeName(nodeNamespace, components.m_localName);
        updateImplWithNamespaceAndElementName(impl, nodeNamespace, nodeName);
        return adoptRef(impl);
    }

    return Ref { impl };
}

Ref<QualifiedName::QualifiedNameImpl> QualifiedNameCache::getOrCreate(const QualifiedNameComponents& components, Namespace nodeNamespace, NodeName nodeName)
{
    auto addResult = m_cache.add<QNameComponentsTranslator>(components);
    auto& impl = **addResult.iterator;

    if (addResult.isNewEntry) {
        updateImplWithNamespaceAndElementName(impl, nodeNamespace, nodeName);
        return adoptRef(impl);
    }

    return Ref { impl };
}

void QualifiedNameCache::remove(QualifiedName::QualifiedNameImpl& impl)
{
    m_cache.remove(&impl);
}

} // namespace WebCore
