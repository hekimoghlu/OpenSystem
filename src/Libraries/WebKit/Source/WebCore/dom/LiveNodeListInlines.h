/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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

#include "CollectionIndexCacheInlines.h"
#include "LiveNodeList.h"

namespace WebCore {

ALWAYS_INLINE bool shouldInvalidateTypeOnAttributeChange(NodeListInvalidationType type, const QualifiedName& attrName)
{
    switch (type) {
    case NodeListInvalidationType::InvalidateOnClassAttrChange:
        return attrName == HTMLNames::classAttr;
    case NodeListInvalidationType::InvalidateOnNameAttrChange:
        return attrName == HTMLNames::nameAttr;
    case NodeListInvalidationType::InvalidateOnIdNameAttrChange:
        return attrName == HTMLNames::idAttr || attrName == HTMLNames::nameAttr;
    case NodeListInvalidationType::InvalidateOnForTypeAttrChange:
        return attrName == HTMLNames::forAttr || attrName == HTMLNames::typeAttr;
    case NodeListInvalidationType::InvalidateForFormControls:
        return attrName == HTMLNames::nameAttr || attrName == HTMLNames::idAttr || attrName == HTMLNames::forAttr
            || attrName == HTMLNames::formAttr || attrName == HTMLNames::typeAttr;
    case NodeListInvalidationType::InvalidateOnHRefAttrChange:
        return attrName == HTMLNames::hrefAttr;
    case NodeListInvalidationType::DoNotInvalidateOnAttributeChanges:
        return false;
    case NodeListInvalidationType::InvalidateOnAnyAttrChange:
        return true;
    }
    return false;
}

ALWAYS_INLINE void LiveNodeList::invalidateCacheForAttribute(const QualifiedName& attributeName) const
{
    if (shouldInvalidateTypeOnAttributeChange(m_invalidationType, attributeName))
        invalidateCache();
}

inline ContainerNode& LiveNodeList::rootNode() const
{
    if (isRootedAtTreeScope() && m_ownerNode->isInTreeScope())
        return m_ownerNode->treeScope().rootNode();
    return m_ownerNode;
}

template <class NodeListType>
unsigned CachedLiveNodeList<NodeListType>::length() const
{
    return m_indexCache.nodeCount(nodeList());
}

template <class NodeListType>
Node* CachedLiveNodeList<NodeListType>::item(unsigned offset) const
{
    return m_indexCache.nodeAt(nodeList(), offset);
}

template <class NodeListType>
auto CachedLiveNodeList<NodeListType>::collectionBegin() const -> Iterator
{
    return Traversal::begin(nodeList(), rootNode());
}

template <class NodeListType>
auto CachedLiveNodeList<NodeListType>::collectionLast() const -> Iterator
{
    return Traversal::last(nodeList(), rootNode());
}

template <class NodeListType>
void CachedLiveNodeList<NodeListType>::collectionTraverseForward(Iterator& current, unsigned count, unsigned& traversedCount) const
{
    Traversal::traverseForward(nodeList(), current, count, traversedCount);
}

template <class NodeListType>
void CachedLiveNodeList<NodeListType>::collectionTraverseBackward(Iterator& current, unsigned count) const
{
    Traversal::traverseBackward(nodeList(), current, count);
}

template <class NodeListType>
bool CachedLiveNodeList<NodeListType>::collectionCanTraverseBackward() const
{
    return true;
}

template <class NodeListType>
void CachedLiveNodeList<NodeListType>::willValidateIndexCache() const
{
    protectedDocument()->registerNodeListForInvalidation(const_cast<CachedLiveNodeList&>(*this));
}

template <class NodeListType>
void CachedLiveNodeList<NodeListType>::invalidateCacheForDocument(Document& document) const
{
    if (m_indexCache.hasValidCache()) {
        document.unregisterNodeListForInvalidation(const_cast<NodeListType&>(nodeList()));
        m_indexCache.invalidate();
    }
}


}
