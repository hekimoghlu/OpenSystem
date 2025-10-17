/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ScrollingTreeStickyNode : public ScrollingTreeNode {
    WTF_MAKE_TZONE_ALLOCATED(ScrollingTreeStickyNode);
public:
    virtual ~ScrollingTreeStickyNode();

    FloatSize scrollDeltaSinceLastCommit() const;

    virtual ScrollingPlatformLayer* layer() const = 0;

protected:
    ScrollingTreeStickyNode(ScrollingTree&, ScrollingNodeID);

    bool commitStateBeforeChildren(const ScrollingStateNode&) override;
    FloatPoint computeLayerPosition() const;
    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const override;

    virtual FloatPoint layerTopLeft() const = 0;

    StickyPositionViewportConstraints m_constraints;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_NODE(ScrollingTreeStickyNode, isStickyNode())

#endif // ENABLE(ASYNC_SCROLLING)
