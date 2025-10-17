/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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

// ScrollingStatePositionedNode is used to manage the layers for z-order composited descendants of overflow:scroll
// which are not containing block descendants (i.e. position:absolute). These layers must have their position inside their ancestor clipping
// layer adjusted on the scrolling thread.
class ScrollingStatePositionedNode final : public ScrollingStateNode {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingStatePositionedNode, WEBCORE_EXPORT);
public:
    template<typename... Args> static Ref<ScrollingStatePositionedNode> create(Args&&... args) { return adoptRef(*new ScrollingStatePositionedNode(std::forward<Args>(args)...)); }

    Ref<ScrollingStateNode> clone(ScrollingStateTree&) override;

    virtual ~ScrollingStatePositionedNode();

    // These are the overflow scrolling nodes whose scroll position affects the layers in this node.
    const Vector<ScrollingNodeID>& relatedOverflowScrollingNodes() const { return m_relatedOverflowScrollingNodes; }
    WEBCORE_EXPORT void setRelatedOverflowScrollingNodes(Vector<ScrollingNodeID>&&);

    WEBCORE_EXPORT void updateConstraints(const AbsolutePositionConstraints&);
    const AbsolutePositionConstraints& layoutConstraints() const { return m_constraints; }

private:
    WEBCORE_EXPORT ScrollingStatePositionedNode(ScrollingNodeID, Vector<Ref<ScrollingStateNode>>&&, OptionSet<ScrollingStateNodeProperty>, std::optional<PlatformLayerIdentifier>, Vector<ScrollingNodeID>&&, AbsolutePositionConstraints&&);
    ScrollingStatePositionedNode(ScrollingStateTree&, ScrollingNodeID);
    ScrollingStatePositionedNode(const ScrollingStatePositionedNode&, ScrollingStateTree&);

    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const final;
    OptionSet<ScrollingStateNode::Property> applicableProperties() const final;

    Vector<ScrollingNodeID> m_relatedOverflowScrollingNodes;
    AbsolutePositionConstraints m_constraints;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_STATE_NODE(ScrollingStatePositionedNode, isPositionedNode())

#endif // ENABLE(ASYNC_SCROLLING)
