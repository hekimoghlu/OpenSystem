/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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

#include "ScrollingConstraints.h"
#include "ScrollingPlatformLayer.h"
#include "ScrollingTreeNode.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ScrollingTreePositionedNode : public ScrollingTreeNode {
    WTF_MAKE_TZONE_ALLOCATED(ScrollingTreePositionedNode);
public:
    virtual ~ScrollingTreePositionedNode();

    const Vector<ScrollingNodeID>& relatedOverflowScrollingNodes() const { return m_relatedOverflowScrollingNodes; }

    FloatSize scrollDeltaSinceLastCommit() const;

    virtual ScrollingPlatformLayer* layer() const = 0;

protected:
    ScrollingTreePositionedNode(ScrollingTree&, ScrollingNodeID);

    bool commitStateBeforeChildren(const ScrollingStateNode&) override;

    WEBCORE_EXPORT void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const override;

    Vector<ScrollingNodeID> m_relatedOverflowScrollingNodes;
    AbsolutePositionConstraints m_constraints;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_NODE(ScrollingTreePositionedNode, isPositionedNode())

#endif // ENABLE(ASYNC_SCROLLING)
