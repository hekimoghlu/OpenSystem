/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
#include "TagCollection.h"

#include "CachedHTMLCollectionInlines.h"
#include "NodeRareDataInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TagCollection);
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TagCollectionNS);
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLTagCollection);

TagCollectionNS::TagCollectionNS(ContainerNode& rootNode, const AtomString& namespaceURI, const AtomString& localName)
    : CachedHTMLCollection(rootNode, CollectionType::ByTag)
    , m_namespaceURI(namespaceURI)
    , m_localName(localName)
{
    ASSERT(m_namespaceURI.isNull() || !m_namespaceURI.isEmpty());
}

TagCollectionNS::~TagCollectionNS()
{
    protectedOwnerNode()->nodeLists()->removeCachedTagCollectionNS(*this, m_namespaceURI, m_localName);
}

TagCollection::TagCollection(ContainerNode& rootNode, const AtomString& qualifiedName)
    : CachedHTMLCollection(rootNode, CollectionType::ByTag)
    , m_qualifiedName(qualifiedName)
{
    ASSERT(qualifiedName != starAtom());
}

TagCollection::~TagCollection()
{
    protectedOwnerNode()->nodeLists()->removeCachedCollection(this, m_qualifiedName);
}

HTMLTagCollection::HTMLTagCollection(ContainerNode& rootNode, const AtomString& qualifiedName)
    : CachedHTMLCollection(rootNode, CollectionType::ByHTMLTag)
    , m_qualifiedName(qualifiedName)
    , m_loweredQualifiedName(qualifiedName.convertToASCIILowercase())
{
    ASSERT(qualifiedName != starAtom());
}

HTMLTagCollection::~HTMLTagCollection()
{
    protectedOwnerNode()->nodeLists()->removeCachedCollection(this, m_qualifiedName);
}

} // namespace WebCore
