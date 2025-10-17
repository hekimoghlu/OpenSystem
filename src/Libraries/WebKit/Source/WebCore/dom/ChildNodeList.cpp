/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
#include "ChildNodeList.h"

#include "CollectionIndexCacheInlines.h"
#include "ElementIterator.h"
#include "NodeRareData.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(EmptyNodeList);
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ChildNodeList);

EmptyNodeList::~EmptyNodeList()
{
    m_owner.get().nodeLists()->removeEmptyChildNodeList(this);
}

ChildNodeList::ChildNodeList(ContainerNode& parent)
    : m_parent(parent)
{
}

ChildNodeList::~ChildNodeList()
{
    Ref { m_parent }->nodeLists()->removeChildNodeList(this);
}

unsigned ChildNodeList::length() const
{
    return m_indexCache.nodeCount(*this);
}

Node* ChildNodeList::item(unsigned index) const
{
    return m_indexCache.nodeAt(*this, index);
}

Node* ChildNodeList::collectionBegin() const
{
    return m_parent->firstChild();
}

Node* ChildNodeList::collectionLast() const
{
    return m_parent->lastChild();
}

void ChildNodeList::collectionTraverseForward(Node*& current, unsigned count, unsigned& traversedCount) const
{
    ASSERT(count);
    for (traversedCount = 0; traversedCount < count; ++traversedCount) {
        current = current->nextSibling();
        if (!current)
            return;
    }
}

void ChildNodeList::collectionTraverseBackward(Node*& current, unsigned count) const
{
    ASSERT(count);
    for (; count && current ; --count)
        current = current->previousSibling();
}

void ChildNodeList::invalidateCache()
{
    m_indexCache.invalidate();
}

} // namespace WebCore
