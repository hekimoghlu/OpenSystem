/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#include "ScrollingStateNode.h"

#include <wtf/Forward.h>

namespace WebCore {

class StickyPositionViewportConstraints;

class ScrollingStateStickyNode final : public ScrollingStateNode {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingStateStickyNode, WEBCORE_EXPORT);
public:
    template<typename... Args> static Ref<ScrollingStateStickyNode> create(Args&&... args) { return adoptRef(*new ScrollingStateStickyNode(std::forward<Args>(args)...)); }

    Ref<ScrollingStateNode> clone(ScrollingStateTree&) override;

    virtual ~ScrollingStateStickyNode();

    WEBCORE_EXPORT void updateConstraints(const StickyPositionViewportConstraints&);
    const StickyPositionViewportConstraints& viewportConstraints() const { return m_constraints; }

private:
    WEBCORE_EXPORT ScrollingStateStickyNode(ScrollingNodeID, Vector<Ref<ScrollingStateNode>>&&, OptionSet<ScrollingStateNodeProperty>, std::optional<PlatformLayerIdentifier>, StickyPositionViewportConstraints&&);
    ScrollingStateStickyNode(ScrollingStateTree&, ScrollingNodeID);
    ScrollingStateStickyNode(const ScrollingStateStickyNode&, ScrollingStateTree&);

    FloatPoint computeLayerPosition(const LayoutRect& viewportRect) const;
    void reconcileLayerPositionForViewportRect(const LayoutRect& viewportRect, ScrollingLayerPositionAction) final;
    FloatSize scrollDeltaSinceLastCommit(const LayoutRect& viewportRect) const;

    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const final;
    OptionSet<ScrollingStateNode::Property> applicableProperties() const final;

    StickyPositionViewportConstraints m_constraints;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_STATE_NODE(ScrollingStateStickyNode, isStickyNode())

#endif // ENABLE(ASYNC_SCROLLING)
