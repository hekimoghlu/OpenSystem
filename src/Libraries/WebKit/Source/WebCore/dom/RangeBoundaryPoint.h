/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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

#include "BoundaryPoint.h"
#include "CharacterData.h"

namespace WebCore {

class RangeBoundaryPoint {
public:
    explicit RangeBoundaryPoint(Node& container);

    Node& container() const;
    Ref<Node> protectedContainer() const { return container(); }
    unsigned offset() const;
    Node* childBefore() const;

    void set(Ref<Node>&& container, unsigned offset, RefPtr<Node>&& childBefore);
    void setOffset(unsigned);

    void setToBeforeNode(Node&);
    void setToAfterNode(Ref<Node>&&);
    void setToBeforeContents(Ref<Node>&&);
    void setToAfterContents(Ref<Node>&&);

    void childBeforeWillBeRemoved();
    void invalidateOffset();

private:
    Ref<Node> m_container;
    unsigned m_offset { 0 };
    RefPtr<Node> m_childBefore;
};

BoundaryPoint makeBoundaryPoint(const RangeBoundaryPoint&);

inline RangeBoundaryPoint::RangeBoundaryPoint(Node& container)
    : m_container(container)
{
}

inline Node& RangeBoundaryPoint::container() const
{
    return m_container;
}

inline Node* RangeBoundaryPoint::childBefore() const
{
    return m_childBefore.get();
}

inline unsigned RangeBoundaryPoint::offset() const
{
    return m_offset;
}

inline void RangeBoundaryPoint::set(Ref<Node>&& container, unsigned offset, RefPtr<Node>&& childBefore)
{
    ASSERT(childBefore == (offset ? container->traverseToChildAt(offset - 1) : nullptr));
    m_container = WTFMove(container);
    m_offset = offset;
    m_childBefore = WTFMove(childBefore);
}

inline void RangeBoundaryPoint::setOffset(unsigned offset)
{
    ASSERT(m_container->isCharacterDataNode());
    ASSERT(m_offset);
    ASSERT(!m_childBefore);
    m_offset = offset;
}

inline void RangeBoundaryPoint::setToBeforeNode(Node& child)
{
    ASSERT(child.parentNode());
    m_container = *child.parentNode();
    m_offset = child.computeNodeIndex();
    m_childBefore = child.previousSibling();
}

inline void RangeBoundaryPoint::setToAfterNode(Ref<Node>&& child)
{
    ASSERT(child->parentNode());
    m_container = *child->parentNode();
    m_offset = child->computeNodeIndex() + 1;
    m_childBefore = WTFMove(child);
}

inline void RangeBoundaryPoint::setToBeforeContents(Ref<Node>&& container)
{
    m_container = WTFMove(container);
    m_offset = 0;
    m_childBefore = nullptr;
}

inline void RangeBoundaryPoint::setToAfterContents(Ref<Node>&& container)
{
    m_container = WTFMove(container);
    m_offset = m_container->length();
    m_childBefore = m_container->lastChild();
}

inline void RangeBoundaryPoint::childBeforeWillBeRemoved()
{
    ASSERT(m_offset);
    ASSERT(m_childBefore);
    --m_offset;
    m_childBefore = m_childBefore->previousSibling();
}

inline void RangeBoundaryPoint::invalidateOffset()
{
    m_offset = m_childBefore ? m_childBefore->computeNodeIndex() + 1 : 0;
}

inline bool operator==(const RangeBoundaryPoint& a, const RangeBoundaryPoint& b)
{
    return &a.container() == &b.container() && a.offset() == b.offset();
}

inline BoundaryPoint makeBoundaryPoint(const RangeBoundaryPoint& point)
{
    return { point.container(), point.offset() };
}

} // namespace WebCore
