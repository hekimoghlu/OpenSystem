/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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

#if ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
#include "ScrollingTreeFixedNode.h"
#include <wtf/RefPtr.h>

namespace WebCore {
class CoordinatedPlatformLayer;

class ScrollingTreeFixedNodeCoordinated final : public ScrollingTreeFixedNode {
public:
    static Ref<ScrollingTreeFixedNodeCoordinated> create(ScrollingTree&, ScrollingNodeID);
    virtual ~ScrollingTreeFixedNodeCoordinated();

private:
    ScrollingTreeFixedNodeCoordinated(ScrollingTree&, ScrollingNodeID);

    CoordinatedPlatformLayer* layer() const override { return m_layer.get(); }

    bool commitStateBeforeChildren(const ScrollingStateNode&) override;
    void applyLayerPositions() override WTF_REQUIRES_LOCK(scrollingTree()->treeLock());

    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const override;

    RefPtr<CoordinatedPlatformLayer> m_layer;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_NODE(ScrollingTreeFixedNodeCoordinated, isFixedNodeCoordinated())

#endif // ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
