/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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

#include "ScrollingTreeFixedNode.h"
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS CALayer;

namespace WebCore {

class ScrollingTreeFixedNodeCocoa : public ScrollingTreeFixedNode {
    WTF_MAKE_TZONE_ALLOCATED(ScrollingTreeFixedNodeCocoa);
public:
    WEBCORE_EXPORT static Ref<ScrollingTreeFixedNodeCocoa> create(ScrollingTree&, ScrollingNodeID);

    virtual ~ScrollingTreeFixedNodeCocoa();

private:
    ScrollingTreeFixedNodeCocoa(ScrollingTree&, ScrollingNodeID);

    CALayer* layer() const final { return m_layer.get(); }

    bool commitStateBeforeChildren(const ScrollingStateNode&) final;
    void applyLayerPositions() final WTF_REQUIRES_LOCK(scrollingTree()->treeLock());

    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const final;

    RetainPtr<CALayer> m_layer;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_NODE(ScrollingTreeFixedNodeCocoa, isFixedNodeCocoa())

#endif // ENABLE(ASYNC_SCROLLING)
