/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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

namespace WebCore {

struct BoundaryPoint {
    Ref<Node> container;
    unsigned offset { 0 };

    BoundaryPoint(Ref<Node>&&, unsigned);

    Document& document() const;
    WEBCORE_EXPORT Ref<Document> protectedDocument() const;
};

bool operator==(const BoundaryPoint&, const BoundaryPoint&);

WTF::TextStream& operator<<(WTF::TextStream&, const BoundaryPoint&);

template<TreeType = Tree> std::partial_ordering treeOrder(const BoundaryPoint&, const BoundaryPoint&);
template<> WEBCORE_EXPORT std::partial_ordering treeOrder<ComposedTree>(const BoundaryPoint&, const BoundaryPoint&);

WEBCORE_EXPORT std::optional<BoundaryPoint> makeBoundaryPointBeforeNode(Node&);
WEBCORE_EXPORT std::optional<BoundaryPoint> makeBoundaryPointAfterNode(Node&);
BoundaryPoint makeBoundaryPointBeforeNodeContents(Node&);
BoundaryPoint makeBoundaryPointAfterNodeContents(Node&);

WEBCORE_EXPORT std::partial_ordering treeOrderForTesting(TreeType, const BoundaryPoint&, const BoundaryPoint&);

inline BoundaryPoint::BoundaryPoint(Ref<Node>&& container, unsigned offset)
    : container(WTFMove(container))
    , offset(offset)
{
}

inline Document& BoundaryPoint::document() const
{
    return container->document();
}

inline bool operator==(const BoundaryPoint& a, const BoundaryPoint& b)
{
    return a.container.ptr() == b.container.ptr() && a.offset == b.offset;
}

inline BoundaryPoint makeBoundaryPointBeforeNodeContents(Node& node)
{
    return { node, 0 };
}

inline BoundaryPoint makeBoundaryPointAfterNodeContents(Node& node)
{
    return { node, node.length() };
}

struct WeakBoundaryPoint {
    WeakPtr<Node, Node::WeakPtrImplType> container;
    unsigned offset { 0 };

    WeakBoundaryPoint(WeakPtr<Node, Node::WeakPtrImplType>&&, unsigned);
};

inline WeakBoundaryPoint::WeakBoundaryPoint(WeakPtr<Node, Node::WeakPtrImplType>&& container, unsigned offset)
    : container(WTFMove(container))
    , offset(offset)
{
}

}
