/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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

#include "ScrollingStateNode.h"

namespace WebCore {

class ScrollingStateOverflowScrollProxyNode : public ScrollingStateNode {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingStateOverflowScrollProxyNode, WEBCORE_EXPORT);
public:
    template<typename... Args> static Ref<ScrollingStateOverflowScrollProxyNode> create(Args&&... args) { return adoptRef(*new ScrollingStateOverflowScrollProxyNode(std::forward<Args>(args)...)); }

    Ref<ScrollingStateNode> clone(ScrollingStateTree&) override;

    virtual ~ScrollingStateOverflowScrollProxyNode();

    // This is the node we get our scroll position from.
    std::optional<ScrollingNodeID> overflowScrollingNode() const { return m_overflowScrollingNodeID; }
    WEBCORE_EXPORT void setOverflowScrollingNode(std::optional<ScrollingNodeID>);

private:
    WEBCORE_EXPORT ScrollingStateOverflowScrollProxyNode(ScrollingNodeID, Vector<Ref<ScrollingStateNode>>&&, OptionSet<ScrollingStateNodeProperty>, std::optional<PlatformLayerIdentifier>, std::optional<ScrollingNodeID> overflowScrollingNode);
    ScrollingStateOverflowScrollProxyNode(ScrollingStateTree&, ScrollingNodeID);
    ScrollingStateOverflowScrollProxyNode(const ScrollingStateOverflowScrollProxyNode&, ScrollingStateTree&);

    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const final;
    OptionSet<ScrollingStateNode::Property> applicableProperties() const final;

    Markable<ScrollingNodeID> m_overflowScrollingNodeID;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_STATE_NODE(ScrollingStateOverflowScrollProxyNode, isOverflowScrollProxyNode())

#endif // ENABLE(ASYNC_SCROLLING)
