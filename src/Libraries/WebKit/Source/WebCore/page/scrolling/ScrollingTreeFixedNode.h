/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#include "ScrollingTree.h"
#include "ScrollingTreeNode.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FixedPositionViewportConstraints;

class ScrollingTreeFixedNode : public ScrollingTreeNode {
    WTF_MAKE_TZONE_ALLOCATED(ScrollingTreeFixedNode);
public:
    virtual ~ScrollingTreeFixedNode();

protected:
    ScrollingTreeFixedNode(ScrollingTree&, ScrollingNodeID);

    virtual ScrollingPlatformLayer* layer() const = 0;

    FloatPoint computeLayerPosition() const;

    bool commitStateBeforeChildren(const ScrollingStateNode&) override;
    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const override;

    FixedPositionViewportConstraints m_constraints;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_NODE(ScrollingTreeFixedNode, isFixedNode())

#endif // ENABLE(ASYNC_SCROLLING)
