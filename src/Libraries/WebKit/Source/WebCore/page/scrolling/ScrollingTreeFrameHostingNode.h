/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollingTreeNode.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ScrollingTree;

class ScrollingTreeFrameHostingNode : public ScrollingTreeNode {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingTreeFrameHostingNode, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT static Ref<ScrollingTreeFrameHostingNode> create(ScrollingTree&, ScrollingNodeID);
    virtual ~ScrollingTreeFrameHostingNode();

    std::optional<LayerHostingContextIdentifier> layerHostingContextIdentifier() const { return m_hostingContext; }
    void setLayerHostingContextIdentifier(std::optional<LayerHostingContextIdentifier>);
    bool isRootOfHostedSubtree() const final { return (bool)m_hostingContext; }

    void willBeDestroyed() override;
    void addHostedChild(RefPtr<ScrollingTreeNode> node) { m_hostedChildren.add(node); }
    void removeHostedChildren();
    void removeHostedChild(RefPtr<ScrollingTreeNode>);

private:
    ScrollingTreeFrameHostingNode(ScrollingTree&, ScrollingNodeID);

    bool commitStateBeforeChildren(const ScrollingStateNode&) final;
    void applyLayerPositions() final;

    WEBCORE_EXPORT void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const override;

    std::optional<LayerHostingContextIdentifier> m_hostingContext;
    UncheckedKeyHashSet<RefPtr<ScrollingTreeNode>> m_hostedChildren;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_NODE(ScrollingTreeFrameHostingNode, isFrameHostingNode())

#endif // ENABLE(ASYNC_SCROLLING)
