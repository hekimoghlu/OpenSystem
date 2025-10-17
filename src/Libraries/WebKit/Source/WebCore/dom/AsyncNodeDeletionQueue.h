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

#include "ContainerNode.h"
#include "Element.h"
#include "HTMLElement.h"
#include "HTMLNames.h"
#include "NodeName.h"

namespace WebCore {

class AsyncNodeDeletionQueue {
public:
    ALWAYS_INLINE void addIfSubtreeSizeIsUnderLimit(NodeVector&& children, unsigned subTreeSize)
    {
        if (m_nodeCount + subTreeSize > s_maxSizeAsyncNodeDeletionQueue)
            return;
        m_nodeCount += subTreeSize;
        m_queue.appendVector(WTFMove(children));
    }

    ALWAYS_INLINE void deleteNodesNow()
    {
        m_queue.clear();
        m_nodeCount = 0;
    }

    ALWAYS_INLINE static ContainerNode::CanDelayNodeDeletion canNodeBeDeletedAsync(const Node& node)
    {
        if (!dynamicDowncast<HTMLElement>(node))
            return ContainerNode::CanDelayNodeDeletion::Yes;
        if (isNodeLikelyLarge(node))
            return ContainerNode::CanDelayNodeDeletion::No;
        return ContainerNode::CanDelayNodeDeletion::Yes;
    }

    ALWAYS_INLINE static bool isNodeLikelyLarge(const Node& node)
    {
        ASSERT(node.isElementNode());

        switch (downcast<Element>(node).elementName()) {
        case NodeName::HTML_audio:
        case NodeName::HTML_body:
        case NodeName::HTML_canvas:
        case NodeName::HTML_iframe:
        case NodeName::HTML_img:
        case NodeName::HTML_object:
        case NodeName::HTML_source:
        case NodeName::HTML_track:
        case NodeName::HTML_video:
        case NodeName::SVG_svg:
            return true;
        default:
            return false;
        }
    }

private:
    Vector<Ref<Node>> m_queue;
    unsigned m_nodeCount { 0 };
    static constexpr unsigned s_maxSizeAsyncNodeDeletionQueue = 100000;
};

} // namespace WebCore
