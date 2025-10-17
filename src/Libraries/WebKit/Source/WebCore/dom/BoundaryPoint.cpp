/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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
#include "BoundaryPoint.h"
#include "ContainerNode.h"
#include "Document.h"

namespace WebCore {

template std::partial_ordering treeOrder<Tree>(const BoundaryPoint&, const BoundaryPoint&);
template std::partial_ordering treeOrder<ShadowIncludingTree>(const BoundaryPoint&, const BoundaryPoint&);

std::optional<BoundaryPoint> makeBoundaryPointBeforeNode(Node& node)
{
    RefPtr parent = node.parentNode();
    if (!parent)
        return std::nullopt;
    return BoundaryPoint { parent.releaseNonNull(), node.computeNodeIndex() };
}

std::optional<BoundaryPoint> makeBoundaryPointAfterNode(Node& node)
{
    RefPtr parent = node.parentNode();
    if (!parent)
        return std::nullopt;
    return BoundaryPoint { parent.releaseNonNull(), node.computeNodeIndex() + 1 };
}

static bool isOffsetBeforeChild(ContainerNode& container, unsigned offset, Node& child)
{
    if (!offset)
        return true;
    // If the container is not the parent, the child is part of a shadow tree, which we sort between offset 0 and offset 1.
    if (child.parentNode() != &container)
        return false;
    unsigned currentOffset = 0;
    for (auto currentChild = container.firstChild(); currentChild && currentChild != &child; currentChild = currentChild->nextSibling()) {
        if (offset <= ++currentOffset)
            return true;
    }
    return false;
}

template<TreeType treeType> std::partial_ordering treeOrderInternal(const BoundaryPoint& a, const BoundaryPoint& b)
{
    if (a.container.ptr() == b.container.ptr())
        return a.offset <=> b.offset;

    for (RefPtr ancestor = b.container.copyRef(); ancestor; ) {
        RefPtr nextAncestor = parent<treeType>(*ancestor);
        if (nextAncestor == a.container.ptr())
            return isOffsetBeforeChild(*nextAncestor, a.offset, *ancestor) ? std::strong_ordering::less : std::strong_ordering::greater;
        ancestor = WTFMove(nextAncestor);
    }

    for (RefPtr ancestor = a.container.copyRef(); ancestor; ) {
        RefPtr nextAncestor = parent<treeType>(*ancestor);
        if (nextAncestor == b.container.ptr())
            return isOffsetBeforeChild(*nextAncestor, b.offset, *ancestor) ? std::strong_ordering::greater : std::strong_ordering::less;
        ancestor = WTFMove(nextAncestor);
    }

    return treeOrder<treeType>(a.container, b.container);
}

template<TreeType treeType> std::partial_ordering treeOrder(const BoundaryPoint& a, const BoundaryPoint& b)
{
    return treeOrderInternal<treeType>(a, b);
}

template<> std::partial_ordering treeOrder<ComposedTree>(const BoundaryPoint& a, const BoundaryPoint& b)
{
    return treeOrderInternal<ComposedTree>(a, b);
}

std::partial_ordering treeOrderForTesting(TreeType type, const BoundaryPoint& a, const BoundaryPoint& b)
{
    switch (type) {
    case Tree:
        return treeOrder<Tree>(a, b);
    case ShadowIncludingTree:
        return treeOrder<ShadowIncludingTree>(a, b);
    case ComposedTree:
        return treeOrder<ComposedTree>(a, b);
    }
    ASSERT_NOT_REACHED();
    return std::partial_ordering::unordered;
}

TextStream& operator<<(TextStream& stream, const BoundaryPoint& boundaryPoint)
{
    TextStream::GroupScope scope(stream);
    stream << "BoundaryPoint ";
    stream.dumpProperty("node", boundaryPoint.container->debugDescription());
    stream.dumpProperty("offset", boundaryPoint.offset);
    return stream;
}

Ref<Document> BoundaryPoint::protectedDocument() const
{
    return document();
}

}
